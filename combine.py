import csv #import csv so i could csvreader but i didnt actually use this
import pandas as pd # pandas to convert my csv to a dataframe
from sklearn.preprocessing import StandardScaler #for normalisation
import numpy as np #
from sklearn.decomposition import PCA # for PCA

expression = pd.read_csv('/home/alanah/Downloads/Expression_Public_22Q4.csv', index_col=0) #read in gene expression csv as a dataframe, index so it reads in row names
mutations = pd.read_csv('/home/alanah/Downloads/Damaging_Mutations.csv', index_col=0)
expression = expression.sort_index(ascending=True)
expression = expression.sort_index(axis=1, ascending=True)
mutations = mutations.sort_index(ascending=True)
print(expression.head())
mutations.replace(0, 'Not Damaging',inplace=True)
mutations.replace(1, 'Damaging',inplace=True)
print(mutations.head())
print(expression.shape)
print(mutations.shape)
frames = [expression,mutations]

result = pd.concat(frames, axis="columns")
#print(result.head())

result = result.dropna()
print(result.shape)
print(result.head())

