
import pandas as pd

df = pd.read_csv('movie_data.csv', encoding='utf-8')
# the following column renaming is necessary on some computers:
# df = df.rename(columns={"review", "1", "sentiment"})
df.head(3)

print(df.head(3))
print(df.shape)

df.loc[0, 'review'][-50:]

# as i can see from here i can get a text that contains html markup and punctuation
print(df.loc[0, 'review'][-50:])

# now we will remove all punctuation marks except emoticon characters and we wil luse python regular expression (regex)

import re

def preprocessor(text):
    text =  re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

print(preprocessor(df.loc[0, 'review'][-50:]))

# testing that it doesn't remove emoticon
print(preprocessor('</a>This :) is :( a test :-)!'))

df['review'] = df['review'].apply(preprocessor)

print('finished!')

def tokenizer(text):
    return text.split()

tokenizer('runners like running and thus they run')

print(tokenizer('runners like running and thus they run'))

# stem words (associate to to the same gambo)

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
tokenizer_porter('runners like running and thus they run')
print(tokenizer_porter('runners like running and thus they run'))

import nltk
nltk.download('stopwords')

# we apply the she stop word sets
from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes'
                             ' running and runs a lot ')
                             if w not in stop]

print([w for w in tokenizer_porter('a runner likes'
                             ' running and runs a lot ')
                             if w not in stop])

                             