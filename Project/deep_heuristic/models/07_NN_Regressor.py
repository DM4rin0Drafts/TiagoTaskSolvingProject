from deep_heuristic.nn_utils import split_data, load_collision, binary_acc, TrainData, TestData
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01

################################################## Data Preprocessing ##################################################

# separate data and labels in 5 different datasets based on direction
x_dataset = np.array([])
y_dataset = np.array([])
z_dataset = np.array([])
nx_dataset = np.array([])
ny_dataset = np.array([])
for item in load_collision():
    item = np.array(item)
    tmp = np.array(item[0])
    if tmp[0] == 0:
        x_dataset = np.append(x_dataset, (tmp[1:5], item[1]))
    elif tmp[0] == 1:
        y_dataset = np.append(y_dataset, (tmp[1:5], item[1]))
    elif tmp[0] == 2:
        z_dataset = np.append(z_dataset, (tmp[1:5], item[1]))
    elif tmp[0] == 3:
        nx_dataset = np.append(nx_dataset, (tmp[1:5], item[1]))
    elif tmp[0] == 4:
        ny_dataset = np.append(ny_dataset, (tmp[1:5], item[1]))

# reshape (input, output)
x_dataset = x_dataset.reshape(-1, 2)
y_dataset = y_dataset.reshape(-1, 2)
z_dataset = z_dataset.reshape(-1, 2)
nx_dataset = nx_dataset.reshape(-1, 2)
ny_dataset = ny_dataset.reshape(-1, 2)

# split each dataset to train and evaluate
x_train_data, x_train_labels, x_eval_data, x_eval_labels = split_data(x_dataset, test_size=0.1, num_of_param=4)
y_train_data, y_train_labels, y_eval_data, y_eval_labels = split_data(y_dataset, test_size=0.1, num_of_param=4)
z_train_data, z_train_labels, z_eval_data, z_eval_labels = split_data(z_dataset, test_size=0.1, num_of_param=4)
nx_train_data, nx_train_labels, nx_eval_data, nx_eval_labels = split_data(nx_dataset, test_size=0.1, num_of_param=4)
ny_train_data, ny_train_labels, ny_eval_data, ny_eval_labels = split_data(ny_dataset, test_size=0.1, num_of_param=4)

# load images
x_train_pics = np.array([x[-8:, 24:40] for x in x_train_data[:, 3]]).reshape(-1, 128)
x_eval_pics = np.array([x[-8:, 24:40] for x in x_eval_data[:, 3]]).reshape(-1, 128)
y_train_pics = [x[16:48, -16:] for x in y_train_data[:, 3]]
z_train_pics = [x[16:48, 16:48] for x in z_train_data[:, 3]]
nx_train_pics = np.array([x[:16, 16:48] for x in nx_train_data[:, 3]]).reshape(-1, 512)
nx_eval_pics = np.array([x[:16, 16:48] for x in nx_eval_data[:, 3]]).reshape(-1, 512)
ny_train_pics = [x[16:48, :16] for x in ny_train_data[:, 3]]

## train data and test data

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train_pics)
X_test = scaler.transform(x_eval_pics)

train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(x_train_labels))
test_data = TestData(torch.FloatTensor(X_test))

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

##################################################### Build Model #####################################################


# Binary classification
class BinaryImageClassification(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryImageClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.layer_out = nn.Linear(int(hidden_size/2), 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(int(hidden_size/2))

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BinaryImageClassification(input_size=128, hidden_size=64).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


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
confusion_matrix(x_eval_labels, y_pred_list)
print(classification_report(x_eval_labels, y_pred_list))
