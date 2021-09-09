from flask import Flask, redirect, url_for, request, render_template, current_app
import trieDict
import deepLearning
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from fuzzywuzzy import fuzz
from memory_profiler import memory_usage
import gc
from threading import Thread
from time import sleep
import concurrent.futures
from pyphonetics import Soundex, RefinedSoundex


app = Flask(__name__)

#model saved
MODEL_PATH = 'model/250000_checkpoint.tar'
WORDS_PATH = 'model/words_alpha.txt'
ERROR_PATH = 'model/wikipedia.dat'

if WORDS_PATH:
	with open(WORDS_PATH) as word_file:
		valid_words = set(word_file.read().split())

	dictionary = list(valid_words)

if ERROR_PATH:
	pairs = []
	with open(ERROR_PATH) as error_file:
		words = error_file.read().split("\n")
	count = 0
	answer = ""
	question = ""
	for idx, word in enumerate(words):
		if (word.startswith("$")):
			answer = word.lower()
			answer = answer.replace("_", " ")
			question = ""
		else:
			question = word.lower()
			question = question.replace("_", " ")

		if len(answer) > 1 and len(question) > 1 :
			pairs.append((question,answer.replace('$','')))

	tuple_pairs = tuple(pairs)
	print("Done")

	pairs = None
	del pairs
	gc.collect()

trie = trieDict.TrieData()
deep = deepLearning.DeepLearning()

@app.route('/')
def home():
	return render_template("index.html", sentence="", suggestions= "")

@app.route("/checkSentence", methods=["POST"])
def checkSentence():
	# add spellchecker
	sentence = request.form.get('sentenceArea')
	sentence = processText(sentence)
	# process sentence here#
	# if sentence is not proper forget the program
	tokens = word_tokenize(sentence)
	incorrects= []

	for token in tokens:
		if not (trie.is_word_in_dictionary(token)):
			if(len(token) > 1):
				incorrects.append(token)

	#now get suggestions
	if (len(incorrects) > 0):
		suggestions = getSuggestions(incorrects)
	else:
		incorrects = 'NO MISSPELLED WORDS'
		suggestions = 'NO SUGGESTIONS'

	return render_template("checkermain.html", check_sentence=True, data={"sentence":sentence, "incorrects": incorrects , "suggestions": suggestions})


def processText(text):

	text = re.sub(r"""
	               [,.;@#?!&$]+
	               \ *           
	               """,
	               " ",
	              text, flags=re.VERBOSE)
	text = re.sub(r'\w*\d\w*', '', text).strip()

	return text


def getSuggestions(incorrects):
	suggestions = {}
	for word in incorrects:
		suggestions[word] = list()

	with concurrent.futures.ThreadPoolExecutor() as executor:
		future_1 = executor.submit(editDistance, suggestions)
		future_2 = executor.submit(dictionarylookup, suggestions)
		# future_3 = executor.submit(deepLearningModel, suggestions)
		suggestions =future_1.result().copy()
		suggestions = future_2.result().copy()
		# suggestions.extend(future_3.result().copy())






	return suggestions

def editDistance(incorrects):
	suggestions = {}

	for i in incorrects.keys():
		suggestions[i] = list()
		for word in dictionary:
			if fuzz.ratio(i.lower(), word.lower()) >= 90:
				suggestions[i].append(word)
			elif fuzz.ratio(i.lower(), word.lower()) >= 70:
				suggestions[i].append(word)

	return suggestions

def dictionarylookup(incorrects):
	suggestions = {}
	for word in incorrects.keys():
		suggestions[word] = list()
		suggestions[word].extend(trie.get_suggestions(word))

	return suggestions

def deepLearningModel(incorrects):
	suggestions = {}
	for word in incorrects.keys():
		suggestions[word] = list()
		suggestions[word].extend(evaluateInput(word).copy);
	return suggestions

def ranking(suggestions, incorrects):
	suggestions = suggestions
	soundex = Soundex()
	rs = RefinedSoundex()
	for word in incorrects.keys():
		suggestions[word][ranking] = list()
		count = 0
		for n in suggestions[word]:
			if (soundex.sounds_like(n,word)):
				pass
	soundex.sounds_like()
	return suggestions

def evaluateInput(word):
	suggestions = []
	suggestions.append(deep.evaluateInput())
	return suggestions

if __name__ == '__main__':
	app.run()
