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
print(mutations.shape)
damaging_list = mutations.apply(lambda row: row.astype(str).str.contains('1').any(), axis=1, result_type='expand') # find all cell lines with mutations
damaging_list = damaging_list.replace({True: 'TRUE', False: 'FALSE'})
damaging = pd.DataFrame (damaging_list, columns = ['Mut'])
print(damaging.head())
expression['Damaging'] = damaging
print(expression.head())
