import numpy as np 
import os
import sys
import torch
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

# N protein x k features matrix, all sequences 700 aa long
print(data.shape)
# (5926, 39900)

# You can reshape it to (N protein x 700 amino acids x 57 features) first. 
data_reshaped = data.reshape(5926, 700, -1)
print(data_reshaped.shape)
# (5926, 700, 57)

# Training split: 
# [0,5430) training
# [5435,5690) test 
# [5690,5926) validation

# The 57 features are:
# [0,22): amino acid residues, with the order of 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq'
# [22,31): Secondary structure labels, with the sequence of 'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq'
# [31,33): N- and C- terminals;
# [33,35): relative and absolute solvent accessibility, used only for training. (absolute accessibility is thresholded at 15; relative accessibility is normalized by the largest accessibility value in a protein and thresholded at 0.15; original solvent accessibility is computed by DSSP)
# [35,57): sequence profile. Note the order of amino acid residues is ACDEFGHIKLMNPQRSTVWXY and it is different from the order for amino acid residues

# The last feature of both amino acid residues and secondary structure labels just mark end of the protein sequence. 
# [22,31) and [33,35) are hidden during testing.

# NoSeq features removed
X_train = data_reshaped[0:5430, :, np.r_[0:21, 31:33, 35:57]]
y_train = data_reshaped[0:5430, :, 22:30]
X_test = data_reshaped[5435:5690, :, np.r_[0:21, 31:33, 35:57]]
y_test = data_reshaped[5435:5690, :, 22:30]
X_val = data_reshaped[5690:5926, :, np.r_[0:21, 31:33, 35:57]]
y_val = data_reshaped[5690:5926, :, 22:30]

print(X_train.shape)
# (5430, 700, 45)       # proteins, aa, features

print(y_train.shape)
# (5430, 700, 8)


###########################
# distribution of classes
###########################

all_labels = data_reshaped[:, :, 22:30]
all_labels = all_labels.reshape(-1, 8)
all_labels = np.argmax(all_labels, axis=1)

Counter(all_labels)
# Counter({0: 3142553, 5: 429890, 2: 270087, 7: 140813, 6: 103017, 3: 48643, 1: 12970, 4: 227})

####################
# Conversion to Q3
####################

# Wikipedia DSSP: These eight types are usually grouped into three larger classes: 
# helix (G, H and I), strand (E and B) and loop (S, T, and C, where C sometimes is represented also as blank space). 

# L(loop) instead of C(coil)?
# [22,31): Secondary structure labels, with the sequence of 'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq'

# L: 22, 28, 29
# S: 23, 24
# H: 25, 26, 27

data_reshaped_Q3 = data_reshaped
data_reshaped_Q3[22] = data_reshaped_Q3[22] + data_reshaped_Q3[28] + data_reshaped_Q3[29]
data_reshaped_Q3[23] = data_reshaped_Q3[23] + data_reshaped_Q3[24] 
data_reshaped_Q3[24] = data_reshaped_Q3[25] + data_reshaped_Q3[26] + data_reshaped_Q3[27]

X_train_Q3 = data_reshaped[0:5430, :, np.r_[0:21, 31:33, 35:57]]
y_train_Q3 = data_reshaped[0:5430, :, 22:25]
X_test_Q3 = data_reshaped[5435:5690, :, np.r_[0:21, 31:33, 35:57]]
y_test_Q3 = data_reshaped[5435:5690, :, 22:25]
X_val_Q3 = data_reshaped[5690:5926, :, np.r_[0:21, 31:33, 35:57]]
y_val_Q3 = data_reshaped[5690:5926, :, 22:25]

all_labels = data_reshaped_Q3[:, :, 22:25]
all_labels = all_labels.reshape(-1, 3)
all_labels = np.argmax(all_labels, axis=1)

print(Counter(all_labels))
# Counter({0: 3865121, 2: 270105, 1: 12974})


###############################################
# pass matrices from DataLoader to conv layers
###############################################

# we have input data in these dimensions -> flag the corners of the matrix to see what happens during loading the data
X_train = torch.ones(5430, 700, 45).float()
y_train = torch.ones(5430, 700, 8)
BATCH_SIZE = 30

X_train[0, 0, 0] = 0
X_train[0, 699, 0] = 2
X_train[0, 0, 44] = 3
X_train[0, 699, 44] = 4

# we use the DataLoader and define a batch size
train = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)

train = None
for i, (sequences, labels) in enumerate(train_loader):
    # within the DataLoader, the data has dimensions (BATCH_SIZE, sequence, channels)
    train = sequences
    print(train.shape)
    break

# the flags are in their correct position
train[0, 0, 0]
train[0, 699, 44]

# for feeding into the model, we need (BATCH_SIZE, channels, sequence) so we transpose the last 2 dimensions
train_trans = torch.transpose(train, 1, 2)
print(train_trans.shape)
print(train_trans[0, 0])
print(train[0, 44])
