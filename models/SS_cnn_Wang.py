import numpy as np 
import os
import sys

import torch
from torch.nn import Module, Sequential, Conv1d, ReLU, MaxPool1d, Linear, CrossEntropyLoss, Dropout
from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from collections import Counter


# Set location of train/test data 
DATA_DIR = os.path.join('..', 'ICML2014')
DATA_FILE = os.path.join(DATA_DIR, 'cullpdb+profile_5926.npy')

# check if file exists and load data
try:
    data = np.load(DATA_FILE)
    # np.savetxt('cullpdb+profile_5926.csv', data, delimiter=',', fmt='%d')
except:
    sys.stderr.write("Error: Cannot open file: {}\n".format(DATA_FILE))
    sys.exit(1)


# Reshape data (see explore_data.py)
data_reshaped = data.reshape(5926, 700, -1)

X_train = data_reshaped[0:5430, :, np.r_[0:22, 31:33, 35:57]]
y_train = data_reshaped[0:5430, :, 22:31]
X_test = data_reshaped[5435:5690, :, np.r_[0:22, 31:33, 35:57]]
y_test = data_reshaped[5435:5690, :, 22:31]
X_val = data_reshaped[5690:5926, :, np.r_[0:22, 31:33, 35:57]]
y_val = data_reshaped[5690:5926, :, 22:31]

print(X_train.shape)
# (5430, 700, 46)       # proteins, aa, features

print(y_train.shape)
# (5430, 700, 9)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train)
X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val)

BATCH_SIZE = 30
# also possible: 181

train = torch.utils.data.TensorDataset(X_train, y_train)
val = torch.utils.data.TensorDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

# Create CNN Model
class CnnModel(Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        # in_channel == number of features (1 in b/w images, 3 in colored images, 45 in our case)
        # out_channel == number of filters to learn
        # kernel_size == window/filter size
        # padding = kernel_size / 2 - 1 to conserve sequence length
        self.conv1 = Conv1d(in_channels=46, out_channels=10, kernel_size=11, padding=5)
        self.conv2 = Conv1d(in_channels=10, out_channels=10, kernel_size=11, padding=5)
        self.conv3 = Conv1d(in_channels=10, out_channels=10, kernel_size=11, padding=5)
        self.conv4 = Conv1d(in_channels=10, out_channels=10, kernel_size=11, padding=5)
        # 9 output categories, including NoSeq class
        self.conv5 = Conv1d(in_channels=10, out_channels=9, kernel_size=11, padding=5)
    def forward(self, x):
        # conv functions need as input: (batchsize, channels, sequence)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        return x


# defining the model
print('creating model')
model = CnnModel()
print(model)

# Hyperparameters
num_epochs = 20
learning_rate = 0.001

# defining optimizer
# optimizer = SGD(model.parameters(), lr=learning_rate)
optimizer = Adam(model.parameters(), lr=learning_rate)

# Cross Entropy Loss
# Calculates weights to tackle class bias
# Weight of class c is the size of largest class divided by the size of class c. 
labels_train = np.argmax(y_train.numpy(), axis=2)       # convert to numpy array for faster performance
# count classes
label_distribution = Counter(labels_train.reshape(-1))

# convert counter object to ordered list
# ignore NoSeq class, will be set to 0
label_distribution = [label_distribution[i] for i in range(len(label_distribution) - 1)]

largest_class = max(label_distribution)

loss_weights = np.fromiter((largest_class / label_distribution[i] for i in range(len(label_distribution))), dtype=float)

loss_weights = np.append(loss_weights, [0])
loss_weights = torch.from_numpy(loss_weights).float()

# print(loss_weights)
# tensor([1.7891e+00, 3.3287e+01, 1.5919e+00, 8.8303e+00, 1.8139e+03, 1.0000e+00, 4.1688e+00, 3.0442e+00, 0.0000e+00])

# loss_weights = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0]).float()
# criterion = CrossEntropyLoss(weight=loss_weights)
criterion = CrossEntropyLoss(ignore_index=8, weight=loss_weights)
# criterion = CrossEntropyLoss()


# Train
print('training model')
for e in range(num_epochs):
    running_loss = 0
    for i, (sequences, labels) in enumerate(train_loader):
        # the sequences of each iterable of the DataLoader object have dimensions (batchsize, 700, 45)
        # but the conv layers need (batchsize, channels, sequence) as input
        # -> transpose second and third dimension
        # print('train shape before transposition:', sequences.shape)
        sequences = torch.transpose(sequences, 1, 2)
        train = Variable(sequences)
        # print('train shape after transposition', train.shape)
        # reverse one-hot encoding to integers 0-7, so that labels are of shape (batchsize, sequence)
        # needed for loss function 
        labels = np.argmax(labels, axis=2)
        labels = Variable(labels)
        # print('labels shape', labels.shape)
        # --------------------------------------------------------
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)
        # Calculate softmax and CEL
        # shape required: (batchsize, number of classes, sequence), (batchsize, sequence)
        loss = criterion(outputs, labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        # 
        running_loss += loss
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(train_loader)))


#########################################################################################
# Evaluation
#########################################################################################

# ############### on training set

# # pass train set through the model
# with torch.no_grad():
#     output = model(torch.transpose(X_train, 1, 2))

# # get predictions
# softmax = torch.exp(output)
# prob = list(softmax.numpy())
# predictions = np.argmax(prob, axis=1)

# # get labels
# labels_train = np.argmax(y_train, axis=2)

# # get overall accuracy
# acc_train = accuracy_score(labels_train.reshape(-1), predictions.reshape(-1))
# print('accuracy train', acc_train)


############### on validation set

# pass val set through model
with torch.no_grad():
    output = model(torch.transpose(X_val, 1, 2))

# get predictions
softmax = torch.exp(output)
prob = list(softmax.numpy())
predictions_val = np.argmax(prob, axis=1)
pred_val_unrolled = predictions_val.reshape(-1)

# get labels
labels_val = np.argmax(y_val, axis=2)
labels_val_unrolled = labels_val.reshape(-1)
labels_val_masked = labels_val_unrolled[labels_val_unrolled != 8]

pred_val_masked = pred_val_unrolled[labels_val_unrolled != 8]

# get overall accuracy
# acc_val = accuracy_score(labels_val_unrolled, pred_val_unrolled)
acc_val = accuracy_score(labels_val_masked, pred_val_masked)
print('accuracy val', acc_val)

# get confusion matrix for all classes (tp, tn, fp, fn)
# print(confusion_matrix(labels_val_unrolled, pred_val_unrolled))
print(confusion_matrix(labels_val_masked, pred_val_masked))

# see precision, recall, f1 score and support for all classes
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7']
print(classification_report(labels_val_masked, pred_val_masked, target_names=target_names))
