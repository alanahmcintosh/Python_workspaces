import pandas as pd
from sklearn.preprocessing import StandardScaler

expression = pd.read_csv('/home/alanah/Downloads/Expression_Public_22Q4.csv', index_col=0) 
mutations = pd.read_csv('/home/alanah/Downloads/Damaging_Mutations.csv', index_col=0)

import plotly.express as px
from sklearn.decomposition import PCA

df = expression
X = df.iloc[:, ].values
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
components = pca.fit_transform(X)

fig = px.scatter(components, x=0, y=1, color=df)
fig.show()



