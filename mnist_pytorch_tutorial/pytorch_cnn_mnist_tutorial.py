#!/usr/bin/env python

# tutorials: 
# https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
# https://www.kaggle.com/sdelecourt/cnn-with-pytorch-for-mnist
# https://medium.com/swlh/pytorch-real-step-by-step-implementation-of-cnn-on-mnist-304b7140605a

import numpy as np
import pandas as pd 

import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Linear, CrossEntropyLoss, Dropout
from torch.optim import SGD
# import torch.nn.functional as F
# import torch.utils.data
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# load data from csv (https://www.kaggle.com/oddrationale/mnist-in-csv?select=mnist_train.csv)
print('loading dataset')
df = pd.read_csv('mnist_train.csv')
print(df.shape)             # (60000, 785)
# print(df.heads())

y = df['label'].values
X = df.drop(['label'], 1).values

print('creating train test split')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_train.shape)      # (48000, 784)
print(X_test.shape)      # (12000, 784)

# for CNNs we actually need the reshaped form and not the flattened
X_train = X_train.reshape(48000, 1, 28, 28)
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train)

X_test = X_test.reshape(12000, 1, 28, 28)
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test)

batch_size = 64

train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

# Create CNN Model
class CnnModel(Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.cnn_layers = Sequential(
            Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0),
            # BatchNorm2d(16), 
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            # BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2)
            # Dropout(0.5, inplace=True)
        )
        self.linear_layers = Sequential(
            # fully connected/dense layer with 10 output nodes
            Linear(32 * 5 * 5, 10)
        )
    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# Hyperparameters
# n_iters = 2500
# num_epochs = int(n_iters / (len(X_train) / batch_size))
num_epochs = 15
learning_rate = 0.001

# defining the model
print('creating model')
model = CnnModel()
# defining optimizer
optimizer = SGD(model.parameters(), lr=learning_rate)

# Cross Entropy Loss
criterion = CrossEntropyLoss()

print(model)

# Train
print('training model')
for e in range(num_epochs):
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # 
        train = Variable(images.view(batch_size, 1, 28, 28))
        labels = Variable(labels)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)
        # Calculate softmax and CEL
        loss = criterion(outputs, labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        # 
        running_loss += loss
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(train_loader)))



# prediction for test set
with torch.no_grad():
    output = model(X_test)

softmax = torch.exp(output)
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on test set
print(accuracy_score(y_test, predictions))
# 0.9775

torch.save(model, './my_cnn_mnist_model.pt')
