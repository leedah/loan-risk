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
        clf_cv = svm.SVC(kernel='linear', C=C,gamma=0.7).fit(X_train_counts,categories)
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
            writer.writerow([id[idx],predLabels[idx]])

# Find Label for the test dataset

def findLabel(df,test_df):
    #count_vect = CountVectorizer(stop_words=stop_words)
    # count_vect.fit(df['Content'])
    # svd = TruncatedSVD(n_components=400)
    # svd.fit(count_vect.transform(df['Content']))
    # X_train_counts = count_vect.transform(df['Content'])
    # X_train_counts = np.add(X_train_counts, count_vect.transform(df['Title']))
    # X_test_counts = count_vect.transform(test_df['Content'])
    # X_test_counts = np.add(X_test_counts, count_vect.transform(test_df['Title']))
    # X_train_counts = svd.transform(X_train_counts)
    # X_test_counts = svd.transform(X_test_counts)

    X_train_counts = df.drop('Label', 1)
    X_test_counts = test_df.drop('Id', 1)

    yPred = classificationMethod('naiveBayes',X_train_counts,X_test_counts,df['Label'],44,44)
    print yPred
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

        # X_train_counts = count_vect.transform(df['Content'].iloc[train_index])
        # X_train_counts = np.add(X_train_counts, count_vect.transform(df['Title'].iloc[train_index])*titleWeight)
        # X_test_counts = count_vect.transform(df['Content'].iloc[test_index])
        # X_test_counts = np.add(X_test_counts, count_vect.transform(df['Title'].iloc[test_index])*titleWeight)
        # X_train_counts = svd.transform(X_train_counts)
        # X_test_counts = svd.transform(X_test_counts)

        print "Fold " + str(fold)
        if method=='ALL':
            runAllClassificationMethods(df,nFolds,X_train_counts,X_test_counts,train_index,test_index)
        else:
            yPred = classificationMethod(method,X_train_counts,X_test_counts,df['Label'].iloc[train_index],train_index,test_index)
            print(classification_report(yPred,df['Label'].iloc[test_index]))
            avgAccuracy+=accuracy_score(df['Label'].iloc[test_index],yPred)
        fold += 1
    if method=='ALL':
        produceStats(nFolds)
    avgAccuracy=avgAccuracy/nFolds
    print "the average accuracy of method "+ method
    print avgAccuracy
    return avgAccuracy


def runAllClassificationMethods(df,nFolds,X_train_counts,X_test_counts,train_index,test_index):
    # classification_method_array=['naiveBayes','RandomForest','SVM']
    classification_method_array=['naiveBayes','RandomForest']
    for idx,method in enumerate(classification_method_array):
        yPred = classificationMethod(method,X_train_counts,X_test_counts,df['Label'].iloc[train_index],train_index,test_index)
        averageAccurracyArray[idx] += accuracy_score(df['Label'].iloc[test_index],yPred)
        print method, averageAccurracyArray[idx]

def writeStats(accuracyArray):
    outputDir = "output/"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    with open('output/EvaluationMetric_10fold.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(['Statistic Measure', 'Naive Bayes','Random Forest','SVM'])
        writer.writerow(['Statistic Measure', 'Naive Bayes','Random Forest'])
        accuracyArray=['Accuracy']+accuracyArray
        writer.writerow(accuracyArray)

def produceStats(nFolds):
    for idx,val in enumerate(averageAccurracyArray):
        averageAccurracyArray[idx]
        averageAccurracyArray[idx] = round(averageAccurracyArray[idx]/nFolds,4)
    writeStats(averageAccurracyArray)


def produceSVMstats(df):
    componentsList = [2,3,4,5,6,10,20,30,40,50,60,70,80,90,100,300,400]  #componentsList = [100,110,120,130]
    accuracyList=[]
    for idx,value in enumerate(componentsList):
        accuracyList.append(crossValidation(df,'SVM',value))
    print accuracyList
    plt.ylim([0.5, 1.0])
    plt.xlim([0.0,120.0])
    plt.xlabel('Components')
    plt.ylabel('Accuracy')
    width = 1
    plt.bar(componentsList,accuracyList, width, color="blue")
    plt.savefig('output/LSIcomponentsAccuracy1')
    plt.show()


# Main

df = pd.read_csv('./dataSets/train.tsv', sep='\t', header=0)
df = df.drop('Id',1) # Do not take into account the id in the classification
df_num = pd.get_dummies(df)
# print df_num
# print "Columns: ", len(df_num.columns)

outputDir = "output/"
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

averageAccurracyArray=[0,0]

# Choose between'ALL','naiveBayes','RandomForest' and 'SVM'
crossValidation(df_num,'ALL',40)  

# Find Labels for testset

testdf =pd.read_csv('dataSets/test.tsv', sep='\t')
test_df_num = pd.get_dummies(testdf)

findLabel(df_num,test_df_num)

print averageAccurracyArray

#produceSVMstats(df)



