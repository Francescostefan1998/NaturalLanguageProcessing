
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

X = df[['Gr Liv Area']].values
y = df['SalePrice'].values
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(

    LinearRegression(),
    max_trials=100, # default value
    min_samples=0.95,
    residual_threshold=None, # default value
    random_state=123
)

ransac.fit(X, y)