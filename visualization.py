import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns

print "Data loading..."

df = pd.read_csv('./dataSets/train.tsv', sep='\t', header=0)
print df.shape

# Create a copy of the Label column replacing 1 and 2 with Good and Bad for plots

label_dict = {1: 'Good',2: 'Bad'}

df_label = df["Label"]

pd.options.mode.chained_assignment = None  # default='warn'

for i in range(len(df_label)):
    df_label[i] = label_dict[df_label[i]]

# Removing Id feature because a sequential feature like this is not good

df = df.drop('Id', 1)

x=df.loc[df['Label'] == 2]

def is_categorical(array_like):
    return array_like.dtype.name == 'object'

sns.set(color_codes=True)

print "Generating plots..."

# Looping on features excluding the 'Label' field

for i, column in enumerate(df.drop('Label', 1)):
    if(is_categorical(df[column])):
        fig=sns.plt.figure(i)
        g = sns.countplot(y=column, hue=df_label, data=df, palette="Set2");
        g.axes.set_title("Histogram for " + column, fontsize=24,alpha=0.5)
        fig.savefig('output/histograms/'+ column);
    else:
        fig=sns.plt.figure(i)
        g = sns.boxplot(y=column, x=df_label, data=df, palette="Set2");
        g.axes.set_title("Box plot for " + column, fontsize=24,alpha=0.5)
        fig.savefig('output/boxPlots/'+ column);

print "Histograms and boxplots generated in the output directory!"







