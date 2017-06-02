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

def informationGain(df,feature):
# Calculate the information gain of a feature in a dataset df

	# Find entropy of the whole dataset
	H_dataset = entropy(df)

	# Find values of feature
	count_values = df[feature].value_counts().to_dict()
	# print "\n\ncount_values\n", count_values
	
	value_sum = 0
	IG = 0

	print "\nFeature ", feature

	# For every row in dataset where feature has the value v
	for v in count_values:
		# print "rows with value", v ,": \n", df[df[feature] == v]
		
		# Find entropy of the rows of the dataset where attribute a has the value v
		H_feature_v = entropy(df[df[feature] == v])
		print "Value ", v, " appeared", count_values[v], "times, Entropy: ", H_feature_v
		
		value_sum += count_values[v]*H_feature_v
		# print "=", value_sum


	IG = H_dataset - value_sum
	print "IG:",IG," = (H_dataset:",H_dataset, ")- (value_sum:",value_sum, ")"
	return IG


# Read dataset

df = pd.read_csv('./dataSets/train_small.tsv', sep='\t', header=0)
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

# IG = informationGain(df,'Attribute2')


# Find information gain of each feature

infogain = dict()

for feature in df_cat.columns:
	infogain[feature] = informationGain(df,feature)

print "\nInformation Gains of all features:"
for feature in sorted(infogain, key=infogain.get):
  print feature, infogain[feature]















