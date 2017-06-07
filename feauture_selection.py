from __future__ import division
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from classification import crossValidation


def entropy(df):
# Calcluate entropy in the dataset given
	
	# Find probabilities of Good and Bad clients
	P = df['Label'].value_counts(normalize = True)

	Hp = 0

	for i in  df['Label'].unique():
		# print "Probability of ", i ,":", P[i] 
		# print "Entropy:", P[i]*math.log(P[i])
		Hp += -P[i]*math.log(P[i])

	return Hp


def informationGain(T,a):
# Calculate the information gain of the attribute a in the dataset T

	# Find entropy of the whole dataset
	H_T =  entropy(T)

	# Find count of values of the attribute a
	values = T[a].value_counts().to_dict()
	value_sum = 0

	for v in values:
		
		# Find wv: the percentage of rows in the dataset where attibute a has the value v
		wv = values[v]/len(T)

		# Find entropy of the rows of the dataset where attribute a has the value v
		H_v = entropy(T[T[a] == v])
		
		value_sum += wv*H_v

	IG = H_T - value_sum 

	# Checks
	# print "\n", a, ":"
	# print "rows with value", v ,": \n", T[T[a] == v]
	# print "values", values[v]
	# print "Value ", v, " wv ", wv, "Entropy: ", H_v
	# print "IG:",IG," = (H_T:",H_T, ")-(value_sum:",value_sum, ")"	
	return IG


def produceRemoveFeaturePlot(df, infogain):
    columnsList=[]
    accuracyList=[]
    my_xticks = []
    
    col = 20
    df_num = pd.get_dummies(df)
    accuracyList.append(crossValidation(df_num,'RandomForest',40))
    columnsList.append(col)

    for feature in sorted(infogain, key=infogain.get):
        df=df.drop(feature,1)
        # print "\nDropping ", feature
        col -= 1
        if(col == 0):
        	break
        df_num = pd.get_dummies(df)
        accuracyList.append(crossValidation(df_num,'RandomForest',40))

        columnsList.append(col)
        my_xticks.append(feature)

    fig = plt.figure()
    fig.canvas.set_window_title('Accuracy plot')

    plt.ylim([0.6, 0.8])
    plt.xlim([21,0])
    plt.title('Average accuracy of Random Forest')
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')  
    plt.xticks(columnsList, columnsList, size='small')
    width = 0.9

    plt.bar(columnsList,accuracyList, width, color="#00cc99")
    plt.show()
    fig.savefig('output/accuracyPlot.png')
    plt.close(fig)


# Read dataset

df = pd.read_csv('./dataSets/train.tsv', sep='\t', header=0)
df = df.drop('Id', 1)
df_cat = df.drop('Label', 1) # Don't calculate information gain of Label
df_i = df

# Convert numerical attibutes to categorical using  5 bins

numericals = [ 'Attribute2', 'Attribute5', 'Attribute8', 'Attribute11', 'Attribute13', 'Attribute16', 'Attribute18' ]

for attr in numericals:
	df[attr] = pd.cut(df[attr],5)
	# print df_cat[attr] 
# print df_cat

# Find information gain of each feature

infogain = dict()

for feature in df_cat.columns:
	infogain[feature] = informationGain(df,feature)

print "\nInformation Gains of all features:\n"
for feature in sorted(infogain, key=infogain.get):
	print '{: <15}'.format(feature),'{:f}'.format(infogain[feature])

produceRemoveFeaturePlot(df_i, infogain)














