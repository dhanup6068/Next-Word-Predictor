import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import load_model
import pickle
import heapq
import warnings
warnings.filterwarnings("ignore")
list1 =[]


#path='D:\\NLP_Project\\nlp.txt'
#text = open(path,encoding="utf8").read().lower()
tokenizer = RegexpTokenizer(r'\w+')
"""words = tokenizer.tokenize(text)

unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

WORD_LENGTH = 5
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])

X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)
for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_words[i]]] = 1"""

WORD_LENGTH = 5

unique_words = np.load('unique_words.npy') 

a_file = open("unique_word_index.pkl", "rb")
unique_word_index = pickle.load(a_file)

model = load_model('keras_next_word_model.h5',compile = False)

def prepare_input(text):
	x = np.zeros((1, WORD_LENGTH,8201))
	for t, word in enumerate(text.split()):
	    x[0, t, unique_word_index[word]] = 1
	return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completions(text, n=3):
    if text == "":
        return("0")
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [unique_words[idx] for idx in next_indices]



