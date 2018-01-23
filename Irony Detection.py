#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:51:55 2017

@author: Mengyun Zheng
"""

#%%
import numpy as np
import pandas as pd
import emoji
from emoji.unicode_codes import UNICODE_EMOJI
import unicodedata
from unidecode import unidecode
from gensim import corpora
import gensim.models as gsm
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.corpus import sentiwordnet as swn
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer, word_tokenize
import enchant
from nltk.corpus import wordnet
from nltk.metrics import edit_distance
np.random.seed(2017)
#%%
replacement_patterns = [
	(r'won\'t', 'will not'),
	(r'can\'t', 'cannot'),
	(r'i\'m', 'i am'),
	(r'ain\'t', 'is not'),
	(r'(\w+)\'ll', '\g<1> will'),
	(r'(\w+)n\'t', '\g<1> not'),
	(r'(\w+)\'ve', '\g<1> have'),
	(r'(\w+)\'s', '\g<1> is'),
	(r'(\w+)\'re', '\g<1> are'),
	(r'(\w+)\'d', '\g<1> would'),
]
#%%
class RegexpReplacer(object):
	""" Replaces regular expression in a text.
	>>> replacer = RegexpReplacer()
	>>> replacer.replace("can't is a contraction")
	'cannot is a contraction'
	>>> replacer.replace("I should've done that thing I didn't do")
	'I should have done that thing I did not do'
	"""
	def __init__(self, patterns=replacement_patterns):
		self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
	
	def replace(self, text):
		s = text
		
		for (pattern, repl) in self.patterns:
			s = re.sub(pattern, repl, s)
		
		return s
#%%
class RepeatReplacer(object):
	""" Removes repeating characters until a valid word is found.
	>>> replacer = RepeatReplacer()
	>>> replacer.replace('looooove')
	'love'
	>>> replacer.replace('oooooh')
	'ooh'
	>>> replacer.replace('goose')
	'goose'
	"""
	def __init__(self):
		self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
		self.repl = r'\1\2\3'

	def replace(self, word):
		if wordnet.synsets(word):
			return word
		
		repl_word = self.repeat_regexp.sub(self.repl, word)
		
		if repl_word != word:
			return self.replace(repl_word)
		else:
			return repl_word
#%%
class SpellingReplacer(object):
	""" Replaces misspelled words with a likely suggestion based on shortest
	edit distance.
	>>> replacer = SpellingReplacer()
	>>> replacer.replace('cookbok')
	'cookbook'
	"""
	def __init__(self, dict_name='en', max_dist=2):
		self.spell_dict = enchant.Dict(dict_name)
		self.max_dist = max_dist
	
	def replace(self, word):
		if self.spell_dict.check(word):
			return word
		
		suggestions = self.spell_dict.suggest(word)
		
		if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
			return suggestions[0]
		else:
			return word
#%%
class AntonymReplacer(object):
	def replace(self, word, pos=None):
		""" Returns the antonym of a word, but only if there is no ambiguity.
		>>> replacer = AntonymReplacer()
		>>> replacer.replace('good')
		>>> replacer.replace('uglify')
		'beautify'
		>>> replacer.replace('beautify')
		'uglify'
		"""
		antonyms = set()
		
		for syn in wordnet.synsets(word, pos=pos):
			for lemma in syn.lemmas():
				for antonym in lemma.antonyms():
					antonyms.add(antonym.name())
		
		if len(antonyms) >= 1:
			return antonyms.pop()
		else:
			return None
	
	def replace_negations(self, sent):
		""" Try to replace negations with antonyms in the tokenized sentence.
		>>> replacer = AntonymReplacer()
		>>> replacer.replace_negations(['do', 'not', 'uglify', 'our', 'code'])
		['do', 'beautify', 'our', 'code']
		>>> replacer.replace_negations(['good', 'is', 'not', 'evil'])
		['good', 'is', 'not', 'evil']
		"""
		i, l = 0, len(sent)
		words = []
		
		while i < l:
			word = sent[i]
			
			if word == 'not' and i+1 < l:
				ant = self.replace(sent[i+1])
				
				if ant:
					words.append(ant)
					i += 2
					continue
			
			words.append(word)
			i += 1
		
		return words
#%%
data = pd.read_table('train_emoji.txt', names=['index', 'label','text'], header=None, delimiter="\t", quoting=3,encoding = "ISO-8859-1")
y = data['label'].values
x = data['text']
#%%
tknzr = TweetTokenizer()
replacer1 = RepeatReplacer()
replacer2 = SpellingReplacer()
replacer3 = RegexpReplacer()
replacer4 = AntonymReplacer()
stop_words = set(stopwords.words('english'))
stop_words.discard("not")
stemmer = SnowballStemmer('english')
emoji_pattern = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
def text_to_words( text ):
    text = str(text)
    text = BeautifulSoup(text,"lxml").get_text()

    #convert emoji into name and append at the end of text
    for char in text:
        if char in emoji.UNICODE_EMOJI:
            text = text + unicodedata.name(char) + " "
     
    #remove emojis from text
    text = re.sub(emoji_pattern, '', text)
    
    

    #remove any url to URL
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text)
    #Convert any @Username to "AT_USER"
    text = re.sub('@[^\s]+','AT_USER',text)
    #Remove additional white spaces
    text = re.sub('[\s]+', ' ', text)
    text = re.sub('[\n]+', ' ', text)
    #Remove not alphanumeric symbols white spaces
    text = re.sub(r'[^\w]', ' ', text)
    #Replace #word with word
    text = re.sub(r'#([^\s]+)', r'\1', text)
    #trim
    text = text.strip('\'"')
    
    replacer3.replace(text)
    
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    
    tokens = tknzr.tokenize(letters_only.lower())
    filtered = [word for word in tokens if word not in stop_words]
    #Removes repeating characters until a valid word is found
    #Replaces misspelled words with a likely suggestion based on shortest edit distance
    replaced = [replacer1.replace(word) for word in filtered]
    replaced = [replacer2.replace(word) for word in replaced]
    replaced = replacer4.replace_negations(replaced)
    
    stemmed = [stemmer.stem(word) for word in replaced] 
    return  stemmed

#%%
size = len(x)
train = []
for i in range(size):
    train.append( text_to_words( x[i] ) )
#%%
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest=train_test_split(train, y, test_size=0.2, random_state=2017)

#%%
# Bag of word
#%%
corpus = []
train_corpus = []
test_corpus = []
for text in train:
    corpus.append(" ".join( text))
for text in xtrain:
    train_corpus.append(" ".join( text))
for text in xtest:
    test_corpus.append(" ".join( text))
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
trainData = vectorizer.fit_transform(corpus)
xtrain1=vectorizer.transform(train_corpus)
xtest1=vectorizer.transform(test_corpus)
xtrain1 = xtrain1.toarray()
xtest1 = xtest1.toarray()
#%%
from sklearn import linear_model
log = linear_model.LogisticRegression(C=1).fit(xtrain1, ytrain)
#%%
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
y_log1 = log.predict(xtest1)
y_log2 = log.predict(xtrain1)
print("Accuracy score is: ",accuracy_score(y_log1, ytest))
print("F1 score is: ",f1_score(y_log1, ytest, average="macro"))
print("precision score is: ", precision_score(y_log1, ytest, average="macro"))
print("recall score is: ", recall_score(y_log1, ytest, average="macro"))
#%%
import xgboost as xgb
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(xtrain1, ytrain)
#%%
y_gbm1 = gbm.predict(xtest1)
y_gbm2 = gbm.predict(xtrain1)
print("Accuracy score is: ",accuracy_score(y_gbm1, ytest))
print("F1 score is: ",f1_score(y_gbm1, ytest, average="macro"))
print("precision score is: ", precision_score(y_gbm1, ytest, average="macro"))
print("recall score is: ", recall_score(y_gbm1, ytest, average="macro"))
#%%
from sklearn import svm
svc = svm.SVC(kernel='linear').fit(xtrain1, ytrain)
#%%
y_svc1 = svc.predict(xtest1)
y_svc2 = svc.predict(xtrain1)
print("Accuracy score is: ",accuracy_score(y_svc1, ytest))
print("F1 score is: ",f1_score(y_svc1, ytest, average="macro"))
print("precision score is: ", precision_score(y_svc1, ytest, average="macro"))
print("recall score is: ", recall_score(y_svc1, ytest, average="macro"))

#%%
# Word2vec
#%%
model = gsm.KeyedVectors.load_word2vec_format('word2vec_twitter_model.bin', binary=True,unicode_errors='ignore')
def makeFeatureVec_w2v(words):

    featureVec = np.zeros((400,),dtype ='float32')    
    nwords = 0
    
    for word in words:
        if word in model.vocab:
            embedding_vector = model[word]
            if embedding_vector is not None: 
                nwords = nwords + 1
                featureVec = np.add(featureVec,embedding_vector)
    # Divide the result by the number of words to get the average
    if nwords != 0:
        featureVec = np.divide(featureVec,nwords)
    return featureVec
#%%
xtrain2 = []
xtest2 = []
for words in xtrain:
    xtrain2.append(makeFeatureVec_w2v(words))
for words in xtest:
    xtest2.append(makeFeatureVec_w2v(words))
xtrain2 = np.asarray(xtrain2)
xtest2 = np.asarray(xtest2)
#%%
from sklearn import linear_model
log = linear_model.LogisticRegression(C=1).fit(xtrain2, ytrain)
#%%
y_log3 = log.predict(xtest2)
y_log4 = log.predict(xtrain2)
print("Accuracy score is: ",accuracy_score(y_log3, ytest))
print("F1 score is: ",f1_score(y_log3, ytest, average="macro"))
print("precision score is: ", precision_score(y_log3, ytest, average="macro"))
print("recall score is: ", recall_score(y_log3, ytest, average="macro"))
#%%
import xgboost as xgb
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=500, learning_rate=0.05).fit(xtrain2, ytrain)
#%%
y_gbm3 = gbm.predict(xtest2)
y_gbm4 = gbm.predict(xtrain2)

print("Accuracy score is: ",accuracy_score(y_gbm3, ytest))
print("F1 score is: ",f1_score(y_gbm3, ytest, average="macro"))
print("precision score is: ", precision_score(y_gbm3, ytest, average="macro"))
print("recall score is: ", recall_score(y_gbm3, ytest, average="macro"))
#%%
svc = svm.LinearSVC(random_state=0).fit(xtrain2, ytrain)
#%%
y_svc3 = svc.predict(xtest2)
y_svc4 = svc.predict(xtrain2)
print("Accuracy score is: ",accuracy_score(y_svc3, ytest))
print("F1 score is: ",f1_score(y_svc3, ytest, average="macro"))
print("precision score is: ", precision_score(y_svc3, ytest, average="macro"))
print("recall score is: ", recall_score(y_svc3, ytest, average="macro"))

#%%
# sentiment score
#%%
def score(words):
    tagged=nltk.pos_tag(words) 
    pscore=0
    nscore=0
    th=0.85
    for i in range(len(tagged)):
         if 'NN' in tagged[i][1] and len(list(swn.senti_synsets(tagged[i][0],'n')))>0:
             if (list(swn.senti_synsets(tagged[i][0],'n'))[0]).obj_score()<th:   
                 pscore+=(list(swn.senti_synsets(tagged[i][0],'n'))[0]).pos_score() 
                 nscore+=(list(swn.senti_synsets(tagged[i][0],'n'))[0]).neg_score()
         elif 'VB' in tagged[i][1] and len(list(swn.senti_synsets(tagged[i][0],'v')))>0:
             if (list(swn.senti_synsets(tagged[i][0],'v'))[0]).obj_score()<th:
                 pscore+=(list(swn.senti_synsets(tagged[i][0],'v'))[0]).pos_score()
                 nscore+=(list(swn.senti_synsets(tagged[i][0],'v'))[0]).neg_score()
         elif 'JJ' in tagged[i][1] and len(list(swn.senti_synsets(tagged[i][0],'a')))>0:
             if (list(swn.senti_synsets(tagged[i][0],'a'))[0]).obj_score()<th:
                 pscore+=(list(swn.senti_synsets(tagged[i][0],'a'))[0]).pos_score()
                 nscore+=(list(swn.senti_synsets(tagged[i][0],'a'))[0]).neg_score()
         elif 'RB' in tagged[i][1] and len(list(swn.senti_synsets(tagged[i][0],'r')))>0:
             if (list(swn.senti_synsets(tagged[i][0],'r'))[0]).obj_score()<th:
                 pscore+=(list(swn.senti_synsets(tagged[i][0],'r'))[0]).pos_score()
                 nscore+=(list(swn.senti_synsets(tagged[i][0],'r'))[0]).neg_score()
    pscore = pscore/len(tagged)
    nscore = nscore/len(tagged)
    return pscore, nscore

#%%
xtrain3=np.zeros((3067,2))
xtest3 = np.zeros((767,2))
for i in range(3067):
    tag=score(xtrain[i])
    xtrain3[i,0]=tag[0]
    xtrain3[i,1]=tag[1]
for i in range(767):
    tag1=score(xtest[i])
    xtest3[i,0]=tag1[0]
    xtest3[i,1]=tag1[1]
#%%
from sklearn import linear_model
log = linear_model.LogisticRegression(C=1).fit(xtrain3, ytrain)
#%%
y_log5 = log.predict(xtest3)
y_log6 = log.predict(xtrain3)
print("Accuracy score is: ",accuracy_score(y_log5, ytest))
print("F1 score is: ",f1_score(y_log5, ytest, average="macro"))
print("precision score is: ", precision_score(y_log5, ytest, average="macro"))
print("recall score is: ", recall_score(y_log5, ytest, average="macro"))
#%%
import xgboost as xgb
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=600, learning_rate=0.05).fit(xtrain3, ytrain)
#%%
y_gbm5 = gbm.predict(xtest3)
y_gbm6 = gbm.predict(xtrain3)
print("Accuracy score is: ",accuracy_score(y_gbm5, ytest))
print("F1 score is: ",f1_score(y_gbm5, ytest, average="macro"))
print("precision score is: ", precision_score(y_gbm5, ytest, average="macro"))
print("recall score is: ", recall_score(y_gbm5, ytest, average="macro"))
#%%
svc = svm.LinearSVC(random_state=0).fit(xtrain3, ytrain)
#%%
y_svc5 = svc.predict(xtest3)
y_svc6 = svc.predict(xtrain3)
print("Accuracy score is: ",accuracy_score(y_svc5, ytest))
print("F1 score is: ",f1_score(y_svc5, ytest, average="macro"))
print("precision score is: ", precision_score(y_svc5, ytest, average="macro"))
print("recall score is: ", recall_score(y_svc5, ytest, average="macro"))   

#%%
# Majority Vote
#%%
pre_test = np.vstack((y_log1,y_log3,y_log5,y_gbm1,y_gbm3,y_gbm5,y_svc1,y_svc3,y_svc5))
sumy = np.sum(pre_test,axis=0)
#%%
y_pre = np.zeros(767)
for i in range(767):
    if sumy[i] >= 5:
        y_pre[i] = 1
print("Accuracy score is: ",accuracy_score(y_pre, ytest))
print("F1 score is: ",f1_score(y_pre, ytest, average="macro"))
print("precision score is: ", precision_score(y_pre, ytest, average="macro"))
print("recall score is: ", recall_score(y_pre, ytest, average="macro"))  





