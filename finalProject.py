
import numpy as np
from scipy import spatial
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
from scipy.sparse import csr_matrix
from sklearn.neural_network import MLPClassifier


def main():

    k = 21

    print('Welcome to Stock Portfolio Builder\n')

    trainHolder = []    #Holds all of the stocks as vectors
    trainTitle = []     #Holds the first row which is the attributes
    trainNames = []     #Holds the names of the stocks, first colum
    trainLabel = []     #Holds the labels of each stock - Revenu Growth
    trainDic = {}
    testDic = {}

    testHolder = []
    testTitle = []
    testNames = []
    testLabel = []

    posLabCount = 0
    negLabCount = 0

    counter = 0
    # 2018 Will be used as our TEST file
    for line in open('2018_Financial_Data.dat', 'r'):
        if counter == 0:
            line = line.split(',')
            line = line[: len(line) - 5]
            testTitle = line
            counter += 1


        else:
            line = line.split(',')

            if (line[2] != ""):
                line = line[: len(line) - 3]
                testNames.append(line[0])
                del line[0]
                for s in range(len(line)):
                    if line[s] == '':
                        line[s] = 0
                    line[s] = float(line[s])
                testLabel.append(line[1])
                if (line[1] > 0):               #
                    posLabCount += 1            # added just to count pos and neg labels
                else:                           #
                    negLabCount += 1            #
                del line[1]
                testHolder.append(line)

    testHolder = np.array(testHolder)

    dicIndex = 0
    tcounter = 0

    uYear = input('What year would you like to TRAIN with (Oldest data is 2014 - Newest data is 2017)?   ')

    for line in open(str(uYear)+'_Financial_Data.dat', 'r'):
        if tcounter == 0:
            line = line.split(',')
            line = line[: len(line) - 5]
            trainTitle = line
            tcounter += 1


        else:
            line = line.split(',')
            if (line[2] != ""):
                line = line[: len(line) - 3]
                trainNames.append(line[0])
                del line[0]
                for s in range(len(line)):
                    if line[s] == '':
                        line[s] = 0
                    line[s] = float(line[s])
                trainLabel.append(line[1])
                trainDic.update({dicIndex: line[1]})
                dicIndex += 1
                del line[1]
                trainHolder.append(line)

    trainHolder = np.array(trainHolder)


    result = KNN(trainHolder,testHolder,trainLabel,testLabel,k)
    accuracyResultKNN = getF1Score(result,testLabel)
    print("K-Nearest Neighbors (F1 score): " + str(accuracyResultKNN))

    resultN = NeuralN(trainHolder, testHolder, trainLabel, testLabel)
    accuracyResultNN = getF1Score(resultN, testLabel)
    print("Neural Network (F1 Score): " + str(accuracyResultNN))2017
    ''

    resultD = DecisionT(trainHolder,testHolder,trainLabel,testLabel)
    accuracyResultDec = getF1Score(resultD,testLabel)
    print("Decision Tree (F1 Score): " + str(accuracyResultDec))

    while True:
        print('\nPlease select which option you would like \n')
        print('     1. Create a Portfolio')
        print('     2. Get result of one specific Stock')
        print('     Enter any other number to Quit\n')

        userInput = input('     Input>  ')

        if userInput == '1' :
            userName = input('\nWhat Username would you like you use for you Portfolio?   ')
            log = open(str(userName) + '.txt', 'w')
            log.write("Welcome " + userName + "!  Here is your personal Portfolio... \n")
            log.write("\n")

            updatePortfolio(resultD,accuracyResultDec,accuracyResultKNN,accuracyResultNN,trainDic,trainNames,trainHolder,trainLabel,testHolder,testNames,testLabel,log,uYear)

            print('\nYour portfolio has been created in file  "' + str(userName) + '.txt"')
        elif userInput == '2':
            userStock = input('\nWhich stock would you like to look up (the input must match any sotck names in 2018 data set)?  ')
            userStock = userStock.upper()
            foundName = ''
            for i in range(len(testNames)):
                if userStock == testNames[i]:
                    if resultD[i] == 1:
                        print('\n' + str(userStock) + ' Will have a POSITIVE growth for the year!!!\n')
                    else:
                        print('\n' + str(userStock) + ' Will have a NEGATIVE growth for the year :( \n')
        else:
            exit(1)




def updatePortfolio(resultD,accuracyResultDec,accuracyResultKNN,accuracyResultNN,trainDic,trainNames,trainHolder,trainLabel,testHolder,testNames,testLabel,log,uYear):

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

    log.write("--------------------------------------------------------------------------------------------------\n")
    log.write("Training data set for this portfolio is " + str(uYear) +  "\n")
    log.write("--------------------------------------------------------------------------------------------------\n")
    log.write("The F1-Score from DecisionTree = " + str(accuracyResultDec))
    log.write("\nThe F1-Score from KNN neighbors = " + str(accuracyResultKNN) + "\n")
    log.write("The F1-Score from NeuralNetwork = " + str(accuracyResultNN) + "\n")
    log.write("--------------------------------------------------------------------------------------------------\n")
    log.write("These are the top 10 highest Revenue Growth stocks up to present day in 2018  \n")
    log.write("--------------------------------------------------------------------------------------------------\n")
    for i in top10:
        log.write(str(trainNames[i]) + " Current Revenue -> " + str(trainHolder[i][0]) + "\n")
    log.write("--------------------------------------------------------------------------------------------------\n")
    log.write("Out of those, these are the stocks that Actually will have Revenue Growth based on OUR predictions! \n")
    log.write("--------------------------------------------------------------------------------------------------\n")
    for i in actualGrowth:
        log.write(str(testNames[i]) + " Current Revenue Growth %" + str(testLabel[i])  + "\n")
    log.write("--------------------------------------------------------------------------------------------------\n")





def getAccuracy(result,testLabel):

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
    ##############################################################
    # converting values in array of labels from original values
    # to:  0 if label was a negative float
    #      1 if label was a positive float
    temp = []
    for t in testLabel:
        if t > 0:
            temp.append(1)
        else:
            temp.append(0)
    ###############################################################

    recall = 0
    precision = 0
    F1 = 0

    TP = 0  # true positives
    FN = 0  # false negatives
    FP = 0  # false positives

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
        if (( i % 50) == 0):
            print('\n' * 100)
            print(str(i) + ' out of ' + str(len(testHolder)) + ' records.')

        tempDistance = {}

        for j in range(len(trainHolder)):
            dd = spatial.distance.euclidean(testHolder[i], trainHolder[j])
            tempDistance.update({j: dd})

        sort = sorted(tempDistance, key=lambda x: tempDistance[x])
        topK = sort[:k]
#        print(topK)

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
