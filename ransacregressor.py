
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
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolor='white', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolor='white', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()