from __future__ import division
import pandas as pd
import math
import os
# from scipy.stats import entropy
import matplotlib.pyplot as plt



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


# def findEntropy(df):
# # Calcluate entropy of the dataset
	
# 	# Find probabilities of Good and Bad clients
# 	P = df['Label'].value_counts(normalize = True)

# 	for i in  df['Label'].unique():
# 		# print "Probability of ", i ,":", P[i] 
# 		# print "Entropy:", P[i]*math.log(P[i])
# 		Hp += entropy(P[i])

# 	return Hp

# 	# Calcualate entropy as: -sum(p * log(p))
# 	return entropy(p)

def informationGain(T,a):
# Calculate the information gain of the attribute a in the dataset T

	# Find entropy of the whole dataset
	H_T =  entropy(T)

	print "\n", a, ":"

	# Find count of values of the attribute a
	values = T[a].value_counts().to_dict()
	value_sum = 0

	for v in values:
		# print "rows with value", v ,": \n", T[T[a] == v]
		print "values", values[v]
		
		# Find wv: the percentage of rows in the dataset where attibute a has the value v
		wv = values[v]/len(T)

		# Find entropy of the rows of the dataset where attribute a has the value v
		H_v = entropy(T[T[a] == v])

		print "Value ", v, " wv ", wv, "Entropy: ", H_v
		
		value_sum += wv*H_v
		# print "=", value_sum

	IG = H_T - value_sum
	print "IG:",IG," = (H_T:",H_T, ") - (value_sum:",value_sum, ")"
	return IG


# Read dataset

df = pd.read_csv('./dataSets/train.tsv', sep='\t', header=0)
df = df.drop('Id', 1)
df_cat = df.drop('Label', 1) # Don't calculate information gain of Label

# Convert numerical attibutes to categorical using  5 bins

numericals = [ 'Attribute2', 'Attribute5', 'Attribute8', 'Attribute11', 'Attribute13', 'Attribute16', 'Attribute18' ]

for attr in numericals:
	df[attr] = pd.cut(df[attr],5)
	# print df_cat[attr] 
# print df_cat


# entr = entropy(df)
# print "Dataset Entropy", entr

# entr = findEntropy(df['Label'])
# print "Dataset  Entropy", entr

# Find information gain of each feature

infogain = dict()

for feature in df_cat.columns:
	infogain[feature] = informationGain(df,feature)

print "\nInformation Gains of all features:\n"
for feature in sorted(infogain, key=infogain.get):
  print '{: <15}'.format(feature),'{:f}'.format(infogain[feature])















