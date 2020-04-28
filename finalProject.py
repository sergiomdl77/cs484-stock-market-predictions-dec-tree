from io import StringIO
import string
import nltk
import re
import random
import operator
import numpy as np
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

    k = 7

    print('Welcome to Stock Portfolio Builder\n')
    # userName = input('What Username would you like you use for you Portfolio\n')
    # log = open('userName', 'w')

    trainHolder = []    #Holds all of the stocks as vectors
    trainTitle = []     #Holds the first row which is the attributes
    trainNames = []     #Holds the names of the stocks, first colum
    trainLabel = []     #Holds the labels of each stock - Revenu Growth

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

    trainHolder = np.array(trainHolder)

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

    testHolder = np.array(testHolder)

    result = KNN(trainHolder,testHolder,trainLabel,k)
    accurecyResult = getAccurecy(result,testLabel)

    print(accurecyResult)

def getAccurecy(result,testLabel):

    temp = []
    acCounter = 0

    for i in testLabel:
        if i > 0:
            temp.append(1)
        else:
            temp.append(0)

    for i in range(len(result)):
        if result[i] == temp[i]:
            acCounter += 1

    acResult = acCounter/len(result)

    return acResult




def KNN(trainHolder,testHolder,trainLabel,k):

    finalResult = []

    for i in range(len(testHolder)):
        print(i)
        tempDistance = {}

        for j in range(len(trainHolder)):
            dd = spatial.distance.cosine(testHolder[i], trainHolder[j])
            tempDistance.update({j: dd})

        sort = sorted(tempDistance, key=lambda x: tempDistance[x])
        topK = sort[:k]
        print(topK)

        prediction = getPrediction(topK,trainLabel)
        finalResult.append(prediction)

    return finalResult

def getPrediction(topK,trainLabel):

    posCounter = 0
    negCounter = 0

    for i in topK:
        if trainLabel[i] > 0:
            posCounter += 1
        else:
            negCounter += 1

    if posCounter > negCounter:
        return 1
    else:
        return 0

if __name__ == '__main__':
    main()