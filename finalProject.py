import pandas as pd
from io import StringIO
import string
import nltk
import re
import random
import operator
import numpy
from textblob import TextBlob
from nltk.sentiment.util import *
from nltk.sentiment import SentimentAnalyzer
from nltk.corpus import subjectivity
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import scipy
from scipy import spatial
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from nltk.corpus import subjectivity
from nltk.classify import NaiveBayesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier




def main():

    trainHolder = []
    trainTitle = []
    testHolder = []
    testTitle = []

    counter = 0
    #2017 Will be used as our TEST file
    for line in open('2017_Financial_Data.dat', 'r'):
       if counter == 0:
           line = line.split(',')
           trainTitle = line
           counter += 1
       else:
           line = line.split(',')
           trainHolder.append(line)


    print(trainTitle)
    print(len(trainTitle))
    print(trainHolder[0])
    print(len(trainHolder[0]))

    tcounter = 0
    #2016 Will be used as our TRAIN file
    for line in open('2016_Financial_Data.dat', 'r'):
        if tcounter == 0:
            line = line.split(',')
            testTitle.append(line)
            tcounter += 1
        else:
            line = line.split(',')
            testHolder.append(line)



if __name__ == '__main__':
    main()