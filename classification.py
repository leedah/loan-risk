import pandas as pd
import os
import csv
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


def classificationMethod(method,X_train_counts,X_test_counts,categories,train_index,test_index):
    yPred=None;
    C = 2.0
    if method == 'naiveBayes':
        clf_cv = GaussianNB().fit(X_train_counts,categories)
    elif method == 'RandomForest':
        clf_cv = RandomForestClassifier(n_estimators=128).fit(X_train_counts,categories)# the best result for random forest
    elif method == 'SVM':
        clf_cv = svm.SVC().fit(X_train_counts,categories)
        #clf_cv = svm.SVC(kernel='linear', C=C,gamma=0.7).fit(X_train_counts,categories)
    yPred = clf_cv.predict(X_test_counts)#after training  try to predict
    return yPred;


def createTestSetCategoryCSV(id,predLabels):
    outputDir = "output/"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    with open('output/testSet_Predictions.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Client_ID','Predicted_Label'])
        for idx,value in enumerate(id):
            print id[idx], ": ", label_dict[predLabels[idx]]
            writer.writerow([id[idx],label_dict[predLabels[idx]]])


# Find Label for the test dataset

def findLabel(df,test_df):

    X_train_counts = df.drop('Label', 1)
    X_test_counts = test_df.drop('Id', 1)

    # Best classifier is RandomForest

    yPred = classificationMethod('RandomForest',X_train_counts,X_test_counts,df['Label'],44,44)
    createTestSetCategoryCSV(test_df['Id'],yPred)

# 10-fold Cross Validation with Accuracy

def crossValidation(df, method, n_components):

    avgAccuracy=0
    nFolds=10
    kf = KFold(n_splits=nFolds)
    fold = 0

    for train_index, test_index in kf.split(df):

        df_data = df.drop('Label', 1)
        X_train_counts = df_data.iloc[train_index]
        X_test_counts  = df_data.iloc[test_index]

        # print "Fold " + str(fold)
        if method=='ALL':
            runAllClassificationMethods(df,nFolds,X_train_counts,X_test_counts,train_index,test_index)
        else:
            yPred = classificationMethod(method,X_train_counts,X_test_counts,df['Label'].iloc[train_index],train_index,test_index)
            # print(classification_report(yPred,df['Label'].iloc[test_index]))
            avgAccuracy+=accuracy_score(df['Label'].iloc[test_index],yPred)
        fold += 1
    print "Average accuracy of "+ method
    if method=='ALL':
        produceStats(nFolds)
        print averageAccurracyArray
    else:
        avgAccuracy=avgAccuracy/nFolds
        print avgAccuracy
    return avgAccuracy


def runAllClassificationMethods(df,nFolds,X_train_counts,X_test_counts,train_index,test_index):
    classification_method_array=['naiveBayes','RandomForest','SVM']
    for idx,method in enumerate(classification_method_array):
        yPred = classificationMethod(method,X_train_counts,X_test_counts,df['Label'].iloc[train_index],train_index,test_index)
        averageAccurracyArray[idx] += accuracy_score(df['Label'].iloc[test_index],yPred)
        print method, accuracy_score(df['Label'].iloc[test_index],yPred)

def writeStats(accuracyArray):
    outputDir = "output/"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    with open('output/EvaluationMetric_10fold.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Statistic Measure', 'Naive Bayes','Random Forest','SVM'])
        accuracyArray=['Accuracy']+accuracyArray
        writer.writerow(accuracyArray)

def produceStats(nFolds):
    for idx,val in enumerate(averageAccurracyArray):
        averageAccurracyArray[idx]
        averageAccurracyArray[idx] = round(averageAccurracyArray[idx]/nFolds,5)
    writeStats(averageAccurracyArray)


# Globals

label_dict = {1: 'Good',2: 'Bad'}

df = pd.read_csv('./dataSets/train.tsv', sep='\t', header=0)
df = df.drop('Id',1) # Do not take into account the id in the classification

# Convert all attributes to numerical

df_num = pd.get_dummies(df)

outputDir = "output/"
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

averageAccurracyArray=[0,0,0]

#  Execute code only when module is run directly, not imported

if __name__ == "__main__":

    print "\nRunning 10-fold cross validation for every classifier.."
    crossValidation(df_num,'ALL', 40)
    # Or choose between naiveBayes','RandomForest' and 'SVM'
    #crossValidation(df_num,'SVM', 40)
    #crossValidation(df_num,'RandomForest', 40)

    # Find Labels for testset

    print "\nFinding labels for test set.."

    testdf =pd.read_csv('dataSets/test.tsv', sep='\t')
    test_df_num = pd.get_dummies(testdf)

    findLabel(df_num,test_df_num)


