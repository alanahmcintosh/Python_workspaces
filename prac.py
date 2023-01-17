import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler #for normalisation
import numpy as np #
from sklearn.decomposition import PCA # for PCA
import matplotlib.pyplot as plt
import seaborn as sns

expression = pd.read_csv('/home/alanah/Downloads/Expression_Public_22Q4.csv', index_col=0)
mutations = pd.read_csv('/home/alanah/Downloads/Damaging_Mutations.csv', index_col=0)
expression = expression.sort_index(ascending=True)
#print(mutations.shape)
damaging_list = mutations.apply(lambda row: row.astype(str).str.contains('1').any(), axis=1, result_type='expand')
damaging_list = damaging_list.replace({True: 'TRUE', False: 'FALSE'})
damaging = pd.DataFrame (damaging_list, columns = ['Mut'])

#print(expression.head())
column_names = list(expression.columns.values) #extracting the column names
row_names = list(expression.index.values)
x = expression.iloc[:, ].values #index all the values from the expression datafram, not sure if this is useful
x = StandardScaler().fit_transform(x) # normalizing the features
print(x.shape) #whats the dimensions of the data
print(np.mean(x),np.std(x)) #check mean and stadard deviation of normalised data
normalised_expression = pd.DataFrame(x, columns=column_names, index=row_names)

pca_expression = PCA(n_components=2) #make a PCA with 2 principal components 
principalComponents_exp = pca_expression.fit_transform(normalised_expression)
principal_exp_Df = pd.DataFrame(data = principalComponents_exp
             , columns = ['principal component 1', 'principal component 2'])
#print(principal_exp_Df.tail())
print('Explained variation per principal component: {}'.format(pca_expression.explained_variance_ratio_))
#principal_exp_Df['Damaging'] = damaging
#ped = pd.DataFrame(data=principal_exp_Df, index=row_names )

row_damaging = list(damaging.index.values)
print(row_damaging[0])
print(row_names[0])

idx = 0
res =[]

for row in row_names:
    if row == row_damaging

extract = damaging['Mut']
#print(extract)
principal_exp_Df = principal_exp_Df.join(extract)
#principal_exp_Df = principal_exp_Df.dropna
#print(principal_exp_Df)
