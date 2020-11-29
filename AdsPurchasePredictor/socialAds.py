import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss

data = pd.read_csv("data.csv")

data = data[['User ID', 'Gender', 'Age','EstimatedSalary', 'Purchased']]
data['Purchased'] = data['Purchased'].astype('int')

X = np.asarray(data[['User ID', 'Gender', 'Age','EstimatedSalary']])
y = np.asarray(data['Purchased'])


replace_map = {'Gender' : {'Male': 0, 'Female': 1}}
data_replace = data.copy()
data_replace.replace(replace_map, inplace=True)

X = np.asarray(data_replace[['User ID', 'Gender', 'Age','EstimatedSalary']])
y = np.asarray(data_replace['Purchased'])
X = preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

LR = LogisticRegression(C=0.09, solver='sag').fit(X_train,y_train)

yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

print (classification_report(y_test, yhat))
print("Log Loss::")
print(log_loss(y_test, yhat_prob))
