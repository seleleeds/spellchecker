# Authored by github at https://gist.github.com/joehillen/5973483
import sys
def TrieData():
	# Python program for insert and search
	# operation in a Trie
	WORDS_PATH = 'model/words_alpha.txt'

	class TrieNode(object):
		def __init__(self):
			self.leaves = dict()
			self.is_string = False


	class PrefixTrie(object):
		def __init__(self):
			self.node = TrieNode()

		def insert(self, word):
			cur = self.node
			for c in word:
				if c not in cur.leaves:
					cur.leaves[c] = TrieNode()
				cur = cur.leaves[c]
			cur.is_string = True

		def search(self, word):
			cur = self.node
			for c in word:
				if c not in cur.leaves:
					return False
				cur = cur.leaves[c]
			return cur.is_string

		def helper_suggest(self, cur, temp_word, word_list):
			# it should recurse all the branches of the cur, add the prefix to it and add to the word list
			if cur.is_string:
				word_list.append(temp_word)
			for c in cur.leaves.keys():
				self.helper_suggest(cur.leaves[c], temp_word+c, word_list)

		def suggest(self, prefix):
			word_list = list()
			cur = self.node
			temp_word = ''
			for c in prefix:
				if c not in cur.leaves:
					break
				cur = cur.leaves[c]
				temp_word += c
			if temp_word == prefix:
				self.helper_suggest(cur, temp_word, word_list)
			return word_list


	class SpellCheck(object):
		def __init__(self, dictionary_file_path=WORDS_PATH):
			self.dictionary = PrefixTrie()
			self.__create_dictionary(file_path=dictionary_file_path)

		def __create_dictionary(self, file_path):
			with open(file=file_path, mode='r') as input_file:
				for word in input_file:
					word = word.strip().lower()
					self.dictionary.insert(word)

		@staticmethod
		def __no_vowels(word):
			vowels = {'a', 'e', 'i', 'o', 'u'}
			for v in vowels:
				if v not in word:
					return False
			return True

		@staticmethod
		def __mixed_casing(word):
			title_case = word.title()
			upper_case = word.upper()
			lower_case = word.lower()
			return word not in {title_case, upper_case, lower_case}

		def __repeating_chars(self, word):
			word = word.lower()
			n = len(word)
			if n <= 2:
				return False
			i = 0
			while i < n-2:
				if word[i] == word[i+1] == word[i+2]:
					return True
				i += 1
			return False

		def conforms(self, word):
			return not self.__mixed_casing(word) and not self.__repeating_chars(word)

		def is_word_in_dictionary(self, word):
			return self.dictionary.search(word.lower())

		def get_suggestions(self, word):
			return self.dictionary.suggest(word.lower())

		def run(self, word):
			suggestions = list()
			word_found = True
			if len(word) <= 2:
				if self.__mixed_casing(word):
					word_found = False
					suggestions = self.get_suggestions(word)
				elif not self.is_word_in_dictionary(word):
					word_found = False
			else:
				if not self.conforms(word):
					word_found = False
					suggestions = self.get_suggestions(word)
				elif self.conforms(word) and not self.is_word_in_dictionary(word):
					word_found = False
			return word_found, suggestions

	trie = SpellCheck()
	return trie

