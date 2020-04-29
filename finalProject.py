from io import StringIO
import string
import nltk
#nltk.download('vader_lexicon')
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
from imblearn.over_sampling import SMOTE
from scipy.sparse import csr_matrix
from nltk.corpus import subjectivity
from nltk.classify import NaiveBayesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

sentiment = SentimentIntensityAnalyzer()


def main():

    k = 21

    print('Welcome to Stock Portfolio Builder\n')
    userName = input('What Username would you like you use for you Portfolio\n')
    log = open(str(userName)+'.txt', 'w')
    log.write("Welcome " + userName + " to your personal PortFolio! \n")

    trainHolder = []    #Holds all of the stocks as vectors
    trainTitle = []     #Holds the first row which is the attributes
    trainNames = []     #Holds the names of the stocks, first colum
    trainLabel = []     #Holds the labels of each stock - Revenu Growth
    trainDic = {}

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
            testTitle = line
            counter += 1
        else:
            line = line.split(',')
            line = line[: len(line) - 3]
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

    dicIndex = 0
    tcounter = 0
    # 2016 Will be used as our TRAIN file
    for line in open('2016_Financial_Data.dat', 'r'):
        if tcounter == 0:
            line = line.split(',')
            line = line[: len(line) - 5]
            trainTitle = line
            tcounter += 1
        else:
            line = line.split(',')
            line = line[: len(line) - 3]
            trainNames.append(line[0])
            del line[0]
            for s in range(len(line)):
                if line[s] == '':
                    line[s] = 0
                line[s] = float(line[s])
            trainLabel.append(line[1])
            trainDic.update({dicIndex:line[1]})
            dicIndex +=1
            del line[1]
            trainHolder.append(line)
    trainHolder = np.array(trainHolder)

    result = KNN(trainHolder,testHolder,trainLabel,testLabel,k)
    accurecyResultKNN = getF1Score(result,testLabel)
    print(accurecyResultKNN)

    resultD = DecisionT(trainHolder,testHolder,trainLabel,testLabel)
    accurecyResultDec = getF1Score(resultD,testLabel)

    resultN = NeuralN(trainHolder, testHolder, trainLabel, testLabel)
    accurecyResultNN = getF1Score(resultN, testLabel)

    updatePortfolio(resultD,accurecyResultDec,accurecyResultKNN,accurecyResultNN,trainDic,trainNames,trainHolder,trainLabel,testHolder,testNames,testLabel,log)

def updatePortfolio(resultD,accurecyResultDec,accurecyResultKNN,accurecyResultNN,trainDic,trainNames,trainHolder,trainLabel,testHolder,testNames,testLabel,log):

    sort = sorted(trainDic, key=lambda x: trainDic[x],reverse=True)
    top10 = sort[:10]
    tempGrowth = []
    actualGrowth = []

    for i in top10:
        for j in range(len(testLabel)):
            if trainNames[i] == testNames[j]:
                tempGrowth.append(j)

    for x in tempGrowth:
        if testLabel[x] > 0:
            actualGrowth.append(x)


    log.write("The F1Score from DecisionTree = " + str(accurecyResultDec))
    log.write("\nThe F1Score from KNN neighbors = " + str(accurecyResultKNN) + "\n")
    log.write("The F1Score from NeuralNetwork = " + str(accurecyResultNN) + "\n")
    log.write("---------------------------------------------------------------------\n")
    log.write("These are the top 10 stocks from 2016 that had the highest Growth\n")
    log.write("---------------------------------------------------------------------\n")
    for i in top10:
        log.write(str(trainNames[i]) + " Current Revenue -> " + str(trainHolder[i][0]) + "\n")
    log.write("---------------------------------------------------------------------\n")
    log.write("Out of those, These are the once that Actually Grew biased on OUR predictions! \n")
    log.write("---------------------------------------------------------------------\n")
    for i in actualGrowth:
        log.write(str(testNames[i]) + " Future Revenue Grown %" + str(testLabel[i]) + " Future Revenue -> " + str(testHolder[i][0]) + "\n")
    log.write("---------------------------------------------------------------------\n")





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

def getF1Score(resultD,testLabel):

    temp = []
    for t in testLabel:
        if t > 0:
            temp.append(1)
        else:
            temp.append(0)

    recall = 0
    precision = 0
    F1 = 0

    TP = 0
    FN = 0
    FP = 0

    for i in range(len(resultD)):
        if resultD[i] == 1 and temp[i] == 1:
            TP += 1
        elif resultD[i] == 0 and temp[i] == 1:
            FN += 1
        elif resultD[i] == 1 and temp[i] == 0:
            FP += 1

    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    F1 = (2*(recall*precision))/(recall+precision)

    return F1

def DecisionT(trainHolder,testHolder,trainLabel,testLabel):

    # Truncated Singular Value Decomposition
    tsvd = TruncatedSVD(n_components=200, algorithm='randomized', n_iter=50, random_state=40)
    # SMOTE variable to deal with imbalence data using svm
    imb = SMOTE(random_state=40)

    temp = []
    for i in trainLabel:
        if i > 0:
            temp.append(1)
        else:
            temp.append(0)

    # Transform the Train data for 2016
    train_sparce = csr_matrix(trainHolder)
    # train_sparce = tsvd.fit(train_sparce, temp).transform(train_sparce)
    # train_sparce, temp = imb.fit_sample(train_sparce, temp)

    # Transform the Test data for 2017
    test_sparce = csr_matrix(testHolder)
    # test_sparce = tsvd.transform(test_sparce)

    clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=53)
    clf.fit(train_sparce, temp)
    return clf.predict(test_sparce)

def NeuralN(trainHolder, testHolder, trainLabel, testLabel):

    # Truncated Singular Value Decomposition
    tsvd = TruncatedSVD(n_components=200, algorithm='randomized', n_iter=50, random_state=40)
    # SMOTE variable to deal with imbalence data using svm
    imb = SMOTE(random_state=40)

    temp = []
    for i in trainLabel:
        if i > 0:
            temp.append(1)
        else:
            temp.append(0)

    # Transform the Train data for 2016
    train_sparce = csr_matrix(trainHolder)
    # train_sparce = tsvd.fit(train_sparce, temp).transform(train_sparce)
    # train_sparce, temp = imb.fit_sample(train_sparce, temp)

    # Transform the Test data for 2017
    test_sparce = csr_matrix(testHolder)
    # test_sparce = tsvd.transform(test_sparce)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=53)
    clf.fit(train_sparce, temp)
    return clf.predict(test_sparce)


def KNN(trainHolder,testHolder,trainLabel,testLabel,k):

    finalResult = []

    for i in range(len(testHolder)):
        print(i)
        tempDistance = {}

        for j in range(len(trainHolder)):
            dd = spatial.distance.euclidean(testHolder[i], trainHolder[j])
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