import torch
import torch.nn as nn
from numpy import genfromtxt
import numpy
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from numpy import genfromtxt

# Read Training Data
X_fileName = "X_fastext_med.csv"
Y_fileName = "Y_med.csv"
load_path = "../Data/Train/"
X_train = genfromtxt(load_path + X_fileName, delimiter=',')
Y_train = genfromtxt(load_path + Y_fileName, delimiter=',')

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.2)

numpy.savetxt('Y_true.csv', Y_test, delimiter=",")
# Convert to torch tensors
X_train_NN = torch.from_numpy(X_train).float()
Y_train_NN = torch.from_numpy(Y_train).long()
Y_train_NN += 2 # Offset by 2 to remove negative value buckets
X_test_NN = torch.from_numpy(X_test).float()

# Create model
# Layers
input_size = X_train.shape[1]
h1 = 2000
h2 = 1000
h3 = 500
h4 = 100
num_classes = 5
model = NeuralNet(input_size, h1, h2, h3, h4, num_classes)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X_train_NN)
    loss = loss_function(output, Y_train_NN)
    loss.backward()
    optimizer.step()
    # print("Epoch: {}, Loss = {}".format(epoch, loss))

# NN results
Y_predicted_NN = model(X_test_NN)
Y_predicted_NN = torch.softmax(Y_predicted_NN, 1)
maxRes, Y_predicted_NN = torch.max(Y_predicted_NN, 1)
Y_predicted_NN -= 2

F1 = metrics.f1_score(Y_test, Y_predicted_NN, average='weighted')
precision = metrics.precision_score(Y_test, Y_predicted_NN, average='weighted')
recall = metrics.recall_score(Y_test, Y_predicted_NN, average='weighted')
print("Recall: {}".format(recall))
print("Precision: {}".format(precision))
print("F1 Score: {}".format(F1))
# count_misclassified = (Y_test != Y_predicted_NN).sum()
# print('Misclassified samples: {}'.format(count_misclassified))

numpy.savetxt('Y_predicted_NN.csv', Y_predicted_NN, delimiter=",")

# Log Reg
model = LogisticRegression(solver = 'lbfgs', multi_class = 'ovr')
model.fit(X_train, Y_train)
Y_predicted_LR = model.predict(X_test)

numpy.savetxt('Y_predicted_LR.csv', Y_predicted_LR, delimiter=",")
# # Save model
# save_path = "model"
# # Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#
# # Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])

# torch.save(model.state_dict(), save_path)