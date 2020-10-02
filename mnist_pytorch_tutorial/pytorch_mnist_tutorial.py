#!/usr/bin/env python

########################################################################
# Tutorial handwritten digit recognition using PyTorch and MNIST dataset
# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
########################################################################

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

PATH_TO_STORE_TRAINSET = '/home/henri/Repos/python_for_data_science/group_project_2nd_struct/mnist_pytorch_tutorial'
PATH_TO_STORE_TESTSET = '/home/henri/Repos/python_for_data_science/group_project_2nd_struct/mnist_pytorch_tutorial'

# 1) transform data so that all images have same dimensions and properties 
transform = transforms.Compose([
                # image to tensor/numbers
                transforms.ToTensor(),
                # normalize tensor with mean and standard deviation
                # image = (image - mean) / std
                # images are black and white so we have only one channel
                transforms.Normalize((0.5), (0.5))
            ])

# download data sets, shuffle and transform them 
trainset = datasets.MNIST(PATH_TO_STORE_TRAINSET, download=True, train=True, transform=transform)
valset = datasets.MNIST(PATH_TO_STORE_TESTSET, download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# 2) explore dataset

# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # shape of tensors
# print(images.shape)     # torch.Size([64, 1, 28, 28])
#                         # 64 images in batch, 1 channel, 28x28 pixel
# print(labels.shape)     # torch.Size([64])

# # display image from training set (first one)
# plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
# plt.show()

# # display more images
# figure = plt.figure()
# num_of_images = 60
# for index in range(1, num_of_images + 1):
#     plt.subplot(6, 10, index)
#     plt.axis('off')
#     plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

# plt.show()

# 3) Build NN

# input layer: 28 * 28 = 784 nodes (images will be flattened before fed into network)
# hidden layer 1: 128 nodes (relu activation)
# hidden layer 2: 64 nodes (relu activation)
# output layer: 10 nodes (softmax activation)
# loss layer (cross-entropy)    ????????

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.LogSoftmax(dim=1)
        )

print(model)

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

# log probabilities ???
logps = model(images)
# calculate NLL loss ???
loss = criterion(logps, labels)

# 5) Training Process

# use torch.optim modul to perform gradient descent and update weights by backpropagation
# in each epoch (iteration over whole training set) training loss will gradually decrease
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader: 
        # Flatten MNIST images into 784 long vector
        images = images.view(images.shape[0], -1)
        # Training pass
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        # This is where th emodel learns by backpropagating 
        loss.backward()
        # And optimizes its weights here 
        optimizer.step()
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))

print("\nTraining Time (in minutes) =", (time() - time0) / 60)
# Training Time (in minutes) = 2.7641278862953187

# 6) Testing and Evaluation

images, labels = next(iter(valloader))

# pass image to model, see what model predicts 
# flatten image
img = images[0].view(1, 784)

# no_grad disables gradient calculation (only forward propagation through NN)
with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
# Predicted Digit = 2


correct_count, all_count = 0, 0

for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if true_label == pred_label:
            correct_count += 1
        all_count += 1

print("Number of images tested =", all_count)
# Number of images tested = 10000
print("\nModel Accuracy =", correct_count / all_count)
# Model Accuracy = 0.9711

# 7) Save model

torch.save(model, './my_mnist_model.pt')
