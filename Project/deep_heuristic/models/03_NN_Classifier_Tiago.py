from shlex import shlex

from deep_heuristic.nn_utils import split_data, load_workspace, binary_acc, TrainData, TestData
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001

################################################## Data Preprocessing ##################################################

train_data, train_labels, eval_data, eval_labels = split_data(
    load_workspace(file_reach="../training_data/reach_tiago.pk"), test_size=0., num_of_param=4, shuffle=True, exclude_feature=None)

INPUTS = len(train_data[0])

scaler = StandardScaler()
X_train = scaler.fit_transform(train_data)
X_test = scaler.transform(eval_data)

## train data
train_data = TrainData(torch.FloatTensor(X_train),
                       torch.FloatTensor(train_labels))

## test data
test_data = TestData(torch.FloatTensor(X_test))

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

##################################################### Build Model #####################################################


# Binary classification
class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(INPUTS, 12)
        self.layer_2 = nn.Linear(12, 4)
        # self.layer_3 = nn.Linear(8, 3)
        self.layer_out = nn.Linear(4, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=.2)
        self.batchnorm1 = nn.BatchNorm1d(12)
        self.batchnorm2 = nn.BatchNorm1d(4)
        # self.batchnorm3 = nn.BatchNorm1d(3)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        # x = self.relu(self.layer_3(x))
        # x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = BinaryClassification()
model.to(device)
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)


##################################################### Train Model #####################################################
model.train()
for e in range(1, EPOCHS + 1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

#################################################### Evaluate Model ####################################################

y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
confusion_matrix(eval_labels, y_pred_list)
print(classification_report(eval_labels, y_pred_list))

torch.save(model.state_dict(), "NN_Tiago_Model.pk")
