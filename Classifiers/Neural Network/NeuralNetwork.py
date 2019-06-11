import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_classes):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, hidden_size3)
        self.layer4 = nn.Linear(hidden_size3, hidden_size4)
        self.layer5 = nn.Linear(hidden_size4, num_classes)

    def forward(self, x):
        h1 = self.layer1(x)
        h1 = self.relu(h1)
        h2 = self.layer2(h1)
        h2 = self.relu(h2)
        h3 = self.layer3(h2)
        h3 = self.relu(h3)
        h4 = self.layer4(h3)
        h4 = self.relu(h4)
        output = self.layer5(h4)
        return output