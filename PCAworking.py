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
principalComponents_exp = pca_expression.fit_transform(x)
principal_exp_Df = pd.DataFrame(data = principalComponents_exp
             , columns = ['principal component 1', 'principal component 2'])
print(principal_exp_Df.tail())
print('Explained variation per principal component: {}'.format(pca_expression.explained_variance_ratio_))
principal_exp_Df['Damaging'] = damaging
print(principalComponents_exp)

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['TRUE', 'FALSE']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = principal_exp_Df['Mut']
    plt.scatter(principal_exp_Df.loc[indicesToKeep, 'principal component 1']
              , principal_exp_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
plt.show()


