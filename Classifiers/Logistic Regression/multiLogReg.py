import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from numpy import genfromtxt
#from joblib import dump

# Reading csv's
X = genfromtxt('X_fastext_med.csv', delimiter=',')
y = genfromtxt('Y_med.csv', delimiter=',')

# Train/test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Building and running logistic regression
model = LogisticRegression(solver = 'lbfgs', multi_class = 'ovr')
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print prediction
print y_test

# Writing model to disk
#dump(model, 'LogRegModel.joblib')

# Evaluation metrics

F1 = metrics.f1_score(y_test, prediction, average='weighted')
precision = metrics.precision_score(y_test, prediction, average='weighted')
recall = metrics.recall_score(y_test, prediction, average='weighted')
print "Recall: {}".format(recall)
print "Precision: {}".format(precision)
print "F1 Score: {}".format(F1)
count_misclassified = (y_test != prediction).sum()
print('Misclassified samples: {}'.format(count_misclassified))
