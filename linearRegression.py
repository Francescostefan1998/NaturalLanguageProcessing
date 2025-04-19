import pandas as pd

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']

df = pd.read_csv('https://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep='\t', usecols=columns)

df.head()

print(df.head())

print(df.shape)

df['Central Air'] = df['Central Air'].map({'N': 0, 'Y':1})

print(df.isnull().sum())

df = df.dropna(axis=0)

print(df.isnull().sum())

import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
# scatterplotmatrix(df.values, figsize=(12, 10), names=df.columns, alpha=0.5)

# plt.tight_layout()
# plt.show()

import numpy as np
from mlxtend.plotting import heatmap

cm = np.correcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
plt.tight_layout()
plt.show()
