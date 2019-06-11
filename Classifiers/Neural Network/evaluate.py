import torch
import torch.nn as nn
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from numpy import genfromtxt


# Load model
model_path = "model"

input_size = 4096
h1 = 2000
h2 = 1000
h3 = 500
h4 = 100
num_classes = 5
model = NeuralNet(input_size, h1, h2, h3, h4, num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Load test data
X_fileName = "X_small_2.csv"
load_path = "../Data/Test/"
X_test = genfromtxt(load_path + X_fileName, delimiter=',')
Y_test = genfromtxt(load_path + Y_fileName, delimiter=',')
X_test = torch.from_numpy(X_test).float()
Y_test = torch.from_numpy(Y_test).long()

# Evaluate Model
Y_predicted = model(X_test)
Y_predicted = torch.softmax(Y_predicted, 1)
maxRes, Y_predicted = torch.max(Y_predicted, 1)
Y_predicted -= 2

# print(sum(Y_predicted - 2 != Y_test))
print(Y_predicted)

F1 = metrics.f1_score(y_test, prediction, average='weighted')
precision = metrics.precision_score(y_test, prediction, average='weighted')
recall = metrics.recall_score(y_test, prediction, average='weighted')
print "Recall: {}".format(recall)
print "Precision: {}".format(precision)
print "F1 Score: {}".format(F1)
count_misclassified = (y_test != prediction).sum()
print('Misclassified samples: {}'.format(count_misclassified))