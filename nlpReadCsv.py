
import pandas as pd

df = pd.read_csv('movie_data.csv', encoding='utf-8')
# the following column renaming is necessary on some computers:
# df = df.rename(columns={"review", "1", "sentiment"})
df.head(3)

print(df.head(3))
print(df.shape)