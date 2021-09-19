import pandas as pd
import re 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

df = pd.read_csv("Corona_NLP_train.csv", parse_dates=["TweetAt"], encoding="latin1")

train_text,val_text,train_label,val_label=train_test_split(df.OriginalTweet,df.Sentiment,test_size=0.15,random_state=42)

tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=3)

logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs', random_state=17, verbose=1)

tfidf_logit_pipeline = Pipeline([('tf_idf', tf_idf), ('logit', logit)])

tfidf_logit_pipeline.fit(train_text, train_label)

pred = tfidf_logit_pipeline.predict(val_text)

accuracy_score(val_label, pred)
