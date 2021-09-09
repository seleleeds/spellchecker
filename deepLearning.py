def DeepLearning():
	import torch
	import torch.nn as nn
	import torchvision.models as models
	from torch.nn.utils.rnn import pack_padded_sequence
	from torch.autograd import Variable
	import torch.nn.functional as F
	import string
	from torch import optim
	# import spacy
	# import spacy
	import numpy as np

	import random
	import math
	import time

	model_name = 'seq2seq_model'
	attn_model = 'dot'
	hidden_size = 500
	encoder_n_layers = 2
	decoder_n_layers = 2
	dropout = 0.2
	batch_size = 64
	device = "cpu"
	# Set checkpoint to load from; set to None if starting from scratch
	loadFilename = "model/250000_checkpoint.tar"
	checkpoint_iter = 4000

	if loadFilename:
		# If loading on same machine the model was trained on
		checkpoint = torch.load(loadFilename)
		# If loading a model trained on GPU to CPU
		#checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
		encoder_sd = checkpoint['en']
		decoder_sd = checkpoint['de']
		encoder_optimizer_sd = checkpoint['en_opt']
		decoder_optimizer_sd = checkpoint['de_opt']
		embedding_sd = checkpoint['embedding']

	all_letters = all_letters = string.ascii_letters + " .,);'(-"
	#Run everything after this
	PAD_token = 0  # Used for padding short sentences
	SOS_token = 1  # Start-of-sentence token
	EOS_token = 2  # End-of-sentence token
	all_letters = string.ascii_letters + " .,);'(-"
	alpha2index = {'☕': 0, "SOS":SOS_token, "EOS":EOS_token}
	index2alpha = {PAD_token: "☕", SOS_token: "SOS", EOS_token: "EOS"}
	for num,let in enumerate(all_letters):
		alpha2index[let] = num+3
		index2alpha[num+3] = let
	num_words = len(alpha2index)

	class EncoderRNN(nn.Module):
		def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
			super(EncoderRNN, self).__init__()
			self.n_layers = n_layers
			self.hidden_size = hidden_size
			self.embedding = embedding
			self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

		def forward(self, input_seq, input_lengths, hidden=None):
			embedded = self.embedding(input_seq)
			packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
			outputs,hidden = self.gru(packed, hidden)
			outputs,_ = nn.utils.rnn.pad_packed_sequence(outputs)
			outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
			return outputs, hidden

	# Luong attention layer
	class Attn(nn.Module):
		def __init__(self, method, hidden_size):
			super(Attn, self).__init__()
			self.method = method
			if self.method not in ['dot', 'general', 'concat']:
				raise ValueError(self.method, "is not an appropriate attention method.")
			self.hidden_size = hidden_size
			if self.method == 'general':
				self.attn = nn.Linear(self.hidden_size, hidden_size)
			elif self.method == 'concat':
				self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
				self.v = nn.Parameter(torch.FloatTensor(hidden_size))

		def dot_score(self, hidden, encoder_output):
			return torch.sum(hidden * encoder_output, dim=2)

		def general_score(self, hidden, encoder_output):
			energy = self.attn(encoder_output)
			return torch.sum(hidden * energy, dim=2)

		def concat_score(self, hidden, encoder_output):
			energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
			return torch.sum(self.v * energy, dim=2)

		def forward(self, hidden, encoder_outputs):
			# Calculate the attention weights (energies) based on the given method
			if self.method == 'general':
				attn_energies = self.general_score(hidden, encoder_outputs)
			elif self.method == 'concat':
				attn_energies = self.concat_score(hidden, encoder_outputs)
			elif self.method == 'dot':
				attn_energies = self.dot_score(hidden, encoder_outputs)

			# Transpose max_length and batch_size dimensions
			attn_energies = attn_energies.t()

			# Return the softmax normalized probability scores (with added dimension)
			return F.softmax(attn_energies, dim=1).unsqueeze(1)

	class LuongAttnDecoderRNN(nn.Module):
		def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers, dropout):
			super(LuongAttnDecoderRNN, self).__init__()

			self.attn_model = attn_model
			self.hidden_size = hidden_size
			self.output_size = output_size
			self.n_layers = n_layers
			self.dropout=dropout

			self.embedding = embedding
			self.embedding_dropout = nn.Dropout(dropout)
			self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=False)
			self.concat = nn.Linear(hidden_size * 2, hidden_size)
			self.out = nn.Linear(self.hidden_size, self.output_size)
			self.attn = Attn(attn_model, hidden_size)


		def forward(self, input_seq, hidden, encoder_outputs):
			#input_seq = input_seq.view(1,-1)
			embedded = self.embedding(input_seq)
			embedded = self.embedding_dropout(embedded)
			rnn_output, hidden = self.gru(embedded, hidden)

			attn_weights = self.attn(rnn_output, encoder_outputs)
			context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
			rnn_output = rnn_output.squeeze(0)
			context = context.squeeze(1)
			concat_input = torch.cat((rnn_output, context), 1)
			concat_output = torch.tanh(self.concat(concat_input))
			output = self.out(concat_output)
			output = F.softmax(output, dim=1)
			return output, hidden

	class GreedySearchDecoder(nn.Module):
		def __init__(self, encoder, decoder):
			super(GreedySearchDecoder, self).__init__()
			self.encoder = encoder
			self.decoder = decoder

		def forward(self, input_seq, input_length, max_length):
			encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

			decoder_hidden = encoder_hidden[:decoder.n_layers]
			decoder_input = torch.ones(1, 1, device=device, dtype=torch.long)*SOS_token

			all_tokens = torch.zeros([0], device=device, dtype=torch.long)
			all_scores = torch.zeros([0], device=device)

			for _ in range(max_length):
				decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
				decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

				all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
				all_scores = torch.cat((all_scores, decoder_scores), dim=0)

				decoder_input = torch.unsqueeze(decoder_input, 0)

			return all_tokens, all_scores

	class Run():
		def __init__(self):
			# Initialize word embeddings
			embedding = nn.Embedding(num_words, hidden_size)
			if loadFilename:
				embedding.load_state_dict(embedding_sd)
			# Initialize encoder & decoder models
			encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
			decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, num_words, decoder_n_layers, dropout)

			if loadFilename:
				encoder.load_state_dict(encoder_sd)
			decoder.load_state_dict(decoder_sd)
			# Use appropriate device
			encoder = encoder.to(device)
			decoder = decoder.to(device)
			print('Models built and ready to go!')

		def indexesFromSentence(self,sentence):
			return [alpha2index[alpha] for alpha in sentence] + [EOS_token]

		def evaluate(self,encoder, decoder, searcher, sentence, max_length=60):
			### Format input sentence as a batch
			# words -> indexes
			indexes_batch = [self.indexesFromSentence(sentence)]
			# Create lengths tensor
			lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
			# Transpose dimensions of batch to match models' expectations
			input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
			# Use appropriate device
			input_batch = input_batch.to(device)
			lengths = lengths.to(device)
			# Decode sentence with searcher
			tokens, scores = searcher(input_batch, lengths, max_length)
			# indexes -> words
			decoded_words = [index2alpha[token.item()] for token in tokens]
			return decoded_words

		def evaluateInput(self, encoder, decoder, searcher, pairr):
			# Get input sentence
			input_sentence = pairr
			# Evaluate sentence
			output_words = self.evaluate(encoder, decoder, searcher, input_sentence)
			# Format and print response sentence
			output_words[:] = [x for x in output_words if not (x == 'PAD')] #
			predicted = ""
			for ip in output_words:
				if ip == "EOS":
					break
				else:
					predicted += ip
			#predicted = predicted.ljust(65)
			#target = pairr[1].ljust(65)
			# target = [alpha2index[let] for let in pairr[1]]
			result = predicted
			predicted = [alpha2index[let] for let in predicted]
			N=65
			# target = (target + N * [''])[:N]
			predicted = (predicted + N * [''])[:N]

			#print("Hello")
			return result


	deep = Run()

	return deep

