import pandas as pd

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']

df = pd.read_csv('https://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep='\t', usecols=columns)

df.head()

print(df.head())