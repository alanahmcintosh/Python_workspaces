import csv #import csv so i could csvreader but i didnt actually use this
import pandas as pd # pandas to convert my csv to a dataframe
from sklearn.preprocessing import StandardScaler #for normalisation
import numpy as np #
from sklearn.decomposition import PCA # for PCA
import matplotlib.pyplot as plt

expression = pd.read_csv('/home/alanah/Downloads/Expression_Public_22Q4.csv', index_col=0) #read in gene expression csv as a dataframe, index so it reads in row names
#result = expression.head(10) #test to see if it read in correctly
#print("First 10 rows of the DataFrame:")
#print(result)
#print(expression.iloc[[0]]) # trying the iloc function to see how it works 
column_names = list(expression.columns.values) #extracting the column names
row_names = list(expression.index.values) #extracting rwo names
#print(row_names)


x = expression.iloc[:, ].values #index all the values from the expression datafram, not sure if this is useful
x = StandardScaler().fit_transform(x) # normalizing the features
print(x.shape) #whats the dimensions of the data
print(np.mean(x),np.std(x)) #check mean and stadard deviation of normalised data
normalised_expression = pd.DataFrame(np.transpose(x))#, columns=column_names, index=row_names) # make a dataframe of the normalised data with same row and column names
#print(normalised_expression.head(10)) #checking the datafram creation worked

pca_expression = PCA(n_components=2) #make a PCA q with 2 principal components 
principalComponents_exp = pca_expression.fit_transform(x)
principal_exp_Df = pd.DataFrame(data = principalComponents_exp
             , columns = ['principal component 1', 'principal component 2'])
print(principal_exp_Df.tail())
print('Explained variation per principal component: {}'.format(pca_expression.explained_variance_ratio_))

plt.scatter(pca_expression.components_[0,:],pca_expression.components_[0,:])
print(pca_expression.components_.shape)
plt.show()



