import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math
from datahandler import DataHandler

class BIMModel():
	"""docstring for BIMModel"""
	def __init__(self):
		print("starting model...")
		print('\n')
		self.data_handler = DataHandler()
		self.data = self.data_handler.data
		self.query_path = self.data_handler.current_directory + "\\HW04\\query.txt"
		self.get_word_list()
		self.get_inverted_index()
		self.generate_rsv_weights()

	def run(self):
		print("running query: please wait...\n")
		query = self.get_query()
		self.search_query(query["query"][0], 10)



	def get_word_list(self):
		print("generating word list: please wait...\n")
		data = self.data		
		#first we get all words
		wordlist = [txt for list in data['text'] for txt in list]
		#get only unique values using set
		wordlist = set(wordlist)
		self.wordlist = list(wordlist)

	def get_inverted_index(self):
		# generating an inverted index of terms
		# here we are going to create a dictionary 
		# with all words as keys and the values are the doc file names containing the word
		print("getting inverted index: please wait...\n")

		data = self.data
		word_dict = defaultdict(set)  
		for i, doc in enumerate(data['text']):
			for word in self.wordlist:
			  if word in doc:
			    word_dict[word].add(data.iloc[i, 0])
		self.word_dict = word_dict


	def generate_rsv_weights(self):
		# get rsv ranking weights
		# generate rsv of each word based on the whole training documents
		# generate rsv of doc based on specific doc provided
		print("generating rsv weights: please wait...\n")
		data = self.data
		word_dict = self.word_dict
		wordlist = self.wordlist
		weights = {}
		for word in wordlist:
			terms_in_relevant_docs = len(self.get_relevant_docs(word_dict[word]))
			prob_of_word = terms_in_relevant_docs/(data.shape[0]+0.5)
			_log = math.log(prob_of_word/(1-prob_of_word)) if prob_of_word > 0 else 0
			weights[word] = self.get_idf(word) + _log
		self.weights = weights
	
	def get_idf(self,word):
		word_dict = self.word_dict
		data = self.data
		return (1 + math.log(data.shape[0] / (len(word_dict[word])))) # add 1 for idf smoothing

	def generate_doc_rsv(self,file_name, query):
		#calculates the rsv of a document based on the query provided
		data = self.data
		weights = self.weights
		result = 0
		doc_list = data[data["file_name"] == file_name]["text"]
		for doc in doc_list:
			for word in doc:
			  if word in query:
			    result += weights[word]
		return result
	def get_relevant_docs(self, docs):
		data = self.data
		files = []
		for file_name in list(docs):
			if int(data[data['file_name'] == file_name]["labels"]) == 1:
				files.append(file_name)
		return files

	def rank_docs(self, query, retrieved_docs):
		# rank docs by query provided
		#retreive all docs containing our word
		#get the rsv score of the retrieved docs
		word_dict = self.word_dict
		results = {
		  'file_name': [],
		  'score': []
		}
		docs = list()
		for word in query:
			docs = [*docs, *list(word_dict[word])]
		for file_name in docs:
			results['file_name'].append(file_name)
			results['score'].append(self.generate_doc_rsv(file_name, query))
		results = pd.DataFrame(results).sort_values('score', ascending=False)
		return results
	
	def search_query(self, search_query, retrieved_docs = 10):
		print("searching query: please wait... \n")
		#preprocess the query first then rank
		#tokenize query
		tokenized_text = []
		for text in [search_query]:
			tokenized_text.append(word_tokenize(text))
		query = list(tokenized_text)[0]
		#lower case
		query = [txt.lower() for txt in query]
		#stop words
		stop_words = set(stopwords.words('english'))
		query = [txt for txt in query if not txt in stop_words]
		#stemming
		porter_stemmer = PorterStemmer()
		query = [porter_stemmer.stem(txt) for txt in query]
		results = self.rank_docs(query, retrieved_docs).head(retrieved_docs)
		print('\n')
		print('\n')
		print('\n')
		print('results are: ')
		data = self.data
		for index, score in results.iterrows():
			file_label = data[data["file_name"] == score['file_name']]["labels"]
			print("RSV {"+score['file_name']+"}     = "+str(round(score['score'], 2)) + 
				"   file_label = " + str(int(file_label)))
	def get_query(self):
		print("getting query: please wait...\n")
		query = pd.read_csv(self.query_path, names=["query"], sep=",", header=None)
		return query

		
