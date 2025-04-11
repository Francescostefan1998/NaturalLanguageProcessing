
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


X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

small_param_grid = [{
    'vect__ngram_range': [(1,1)],
    'vect__stop_words': [None],
    'vect__tokenizer': [tokenizer, tokenizer_porter],
    'clf__C': [1.0, 10.0]
},
{
    'vect__ngram_range': [(1,1)],
    'vect__stop_words': [stop, None],
    'vect__tokenizer': [tokenizer],
    'vect__use_idf':[False],
    'vect__norm': [None],
    'clf__penalty': ['l2'],
    'clf__C': [1.0, 10.0]
}
]

lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(solver='liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)

import joblib

joblib.dump(gs_lr_tfidf, 'gs_lr_tfidf_model.pkl')


import joblib

gs_lr_tfidf = joblib.load('gs_lr_tfidf_model.pkl')

# Now you can use it:
print(gs_lr_tfidf.best_params_)
print(gs_lr_tfidf.best_score_)
best_model = gs_lr_tfidf.best_estimator_
print(f'Best parameter set: {gs_lr_tfidf.best_params_}')

print(f'CV Accuracy: {gs_lr_tfidf.best_score_:.3f}')

clf = gs_lr_tfidf.best_estimator_
print(f'Test Accuracy: {clf.score(X_test, y_test):.3f}')