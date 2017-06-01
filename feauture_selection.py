import pandas as pd
import math
import os
from scipy.stats import entropy
import matplotlib.pyplot as plt


def findEntropy(df):
# Calcluate entropy of an attribute 
	
	# Find probabilities of the values of the attribute
	p = df.value_counts()/len(df)
	
	# Calcualate entropy as: -sum(p * log(p)).Entropy is defined as -sum(p * log(p) * p) though! (?)
	return entropy(p)



# Read dataset

df = pd.read_csv('./dataSets/train.tsv', sep='\t', header=0)
df_cat = df.drop('Id', 1)
df_cat = df_cat.drop('Label', 1) # Don't calculate entropy of Label

# Convert numerical attibutes to categorical using  5 bins

numericals = [ 'Attribute2', 'Attribute5', 'Attribute8', 'Attribute11', 'Attribute13', 'Attribute16', 'Attribute18' ]

for attr in numericals:
	df_cat[attr] = pd.cut(df_cat[attr],5)
# 	print df_cat[attr] 
# print df_cat


# Find entropy of each attribute

for feature in df_cat.columns:
	entr = findEntropy(df_cat[feature])
	print df_cat[feature].name, ": Entropy", entr












