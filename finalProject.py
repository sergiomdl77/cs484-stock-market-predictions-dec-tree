from io import StringIO
import string
import nltk
import re
import random
import operator
import numpy
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

    print('Welcome to Stock Portfolio Builder\n')
    # userName = input('What Username would you like you use for you Portfolio\n')
    # log = open('userName', 'w')

    trainHolder = []    #Holds all of the stocks as vectors
    trainTitle = []     #Holds the first row which is the attributes
    trainNames = []     #Holds the names of the stocks, first colum
    trainLabel = []     #Holds the labels of each stock

    testHolder = []
    testTitle = []
    testNames = []
    testLabel = []

    counter = 0
    # 2017 Will be used as our TEST file
    for line in open('2017_Financial_Data.dat', 'r'):
        if counter == 0:
            line = line.split(',')
            line = line[: len(line) - 5]
            trainTitle = line
            counter += 1
        else:
            line = line.split(',')
            line = line[: len(line) - 4]
            trainNames.append(line[0])
            del line[0]
            for s in range(len(line)):
                if line[s] == '':
                    line[s] = 0
                line[s] = float(line[s])
            trainLabel.append(line[1])
            del line[1]
            trainHolder.append(line)

    print(trainLabel)

    tcounter = 0
    # 2016 Will be used as our TRAIN file
    for line in open('2016_Financial_Data.dat', 'r'):
        if tcounter == 0:
            line = line.split(',')
            line = line[: len(line) - 5]
            testTitle = line
            tcounter += 1
        else:
            line = line.split(',')
            line = line[: len(line) - 4]
            testNames.append(line[0])
            del line[0]
            for s in range(len(line)):
                if line[s] == '':
                    line[s] = 0
                line[s] = float(line[s])
            testLabel.append(line[1])
            del line[1]
            testHolder.append(line)


if __name__ == '__main__':
    main()