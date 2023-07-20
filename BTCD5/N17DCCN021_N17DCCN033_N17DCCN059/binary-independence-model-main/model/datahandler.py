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
import nltk
import os

class DataHandler():
	"""docstring for DataHandler"""
	def __init__(self):
		# param limiter => used to control number of training docs being loaded because of memory overload
		self.limiter = limiter
		self.current_directory = os.path.dirname(os.path.realpath(__file__))
		self.path = self.current_directory + "\\HW04\\documents"
		self.labels_path = self.current_directory + "\\HW04\\file_label.txt"
		nltk.download('stopwords')
		nltk.download('punkt')
		self.load_dataset()
		self.load_labels()
		self.get_data_and_label()
		self.preprocess_data()

	def load_dataset(self):
		results = defaultdict(list)
		print('\n')
		print("loading dataset: please wait...\n")
		for file in Path(self.path).iterdir():
			with open(file, "r") as file_open:
			    results["file_name"].append(file.name)
			    results["text"].append(file_open.read())
		data = pd.DataFrame(results)
		#removing .txt extension
		data['file_name'] = data['file_name'].apply(lambda name: name.split(".")[0])
		self.data = data

	def load_labels(self):		
		#loading labels
		print("loading labels: please wait...\n")
		labels = pd.read_csv(self.labels_path, names=["file_name", "label"], sep=",", header=None)
		labels["label"].fillna(0, inplace=True)
		labels = labels.reindex(index=labels.index[::-1])
		self.labels = labels

	def get_data_and_label(self):
		# concat data and labels
		data = self.data	
		labels = self.labels
		data["labels"] = ""
		for file_name in data["file_name"]:
		  index = data[data["file_name"] == file_name].index.values[0]
		  label = labels[labels["file_name"] == file_name]["label"]
		  data["labels"][index] = label.values[0] if len(label) > 0 else 0
		self.data = data

	def preprocess_data(self):
		print("pre-processing data: please wait...\n")
		# tokenize
		data = self.data
		tokenized_text = []
		for text in data["text"]:
			tokenized_text.append(word_tokenize(text))
		data["text"] = list(tokenized_text)
		# lower case
		data['text'] = data['text'].apply(lambda list: [txt.lower() for txt in list])
		# stop word removal
		stop_words = set(stopwords.words('english'))
		data['text'] = data['text'].apply(lambda list: [txt for txt in list if not txt in stop_words])
		# stemming
		porter_stemmer = PorterStemmer()
		data['text'] = data['text'].apply(lambda list: [porter_stemmer.stem(txt) for txt in list])
		self.data = data
