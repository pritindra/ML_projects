import itertools
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from urllib.request import urlretrieve
from sklearn import preprocessing
import numpy as np
import pandas as pd

# Retrieving data
iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urlretrieve(iris)

# Pandas read csv dataset
df = pd.read_csv(iris, sep=',')
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df.columns = attributes

# Normalizing
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
y = df['class'].values

# preprocessing the data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Splitting the dataset to train and test data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# KNN model
k = 10
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

# Prediction
yhat = neigh.predict(X_test)

print(y_test[0:5])
print(yhat[0:5])
