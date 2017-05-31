
import pandas as pd

import random
import matplotlib.pyplot as plt
import seaborn as sns

print "*****************  DATA LOADING *********************"
df = pd.read_csv('./dataSets/train.tsv', sep='\t', header=0)

# Removing Id feature because a sequential feature like this is not good.
df = df.drop('Id', 1)

x=df.loc[df['Label'] == 2]
print x.shape

def is_categorical(array_like):
    return array_like.dtype.name == 'object'

sns.set(color_codes=True)

print "********************   GENERATE HISTOGRAMS AND BOX PLOTS *******************"
# Looping on features excluding the 'Label' field
for i, column in enumerate(df.drop('Label', 1)):
    if(is_categorical(df[column])):
        fig=sns.plt.figure(i)
        g = sns.countplot(y=column, hue="Label", data=df, palette="Reds_d");
        g.axes.set_title("Histogram for " + column, fontsize=24,alpha=0.5)
        fig.savefig('output/histograms/'+ column);
    else:
        fig=sns.plt.figure(i)
        sns.boxplot(y=column, x="Label", data=df);
        g = sns.boxplot(y=column, x="Label", data=df);
        g.axes.set_title("Box plot for " + column, fontsize=24,alpha=0.5)
        fig.savefig('output/boxPlots/'+ column);
