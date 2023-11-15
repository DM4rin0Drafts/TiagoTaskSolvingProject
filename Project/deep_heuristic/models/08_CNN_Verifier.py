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

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001

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
    if tmp[3] < 0.14:
        continue
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
x_train_data, x_train_labels, x_eval_data, x_eval_labels = split_data(x_dataset, test_size=0.99, num_of_param=4)
y_train_data, y_train_labels, y_eval_data, y_eval_labels = split_data(y_dataset, test_size=0.99, num_of_param=4)
z_train_data, z_train_labels, z_eval_data, z_eval_labels = split_data(z_dataset, test_size=0.99, num_of_param=4)
nx_train_data, nx_train_labels, nx_eval_data, nx_eval_labels = split_data(nx_dataset, test_size=0.99, num_of_param=4)
ny_train_data, ny_train_labels, ny_eval_data, ny_eval_labels = split_data(ny_dataset, test_size=0.99, num_of_param=4)

# load images
x_train_pics = np.array([x[-8:, 16:48] for x in x_train_data[:, 3]]).reshape(-1, 256)
x_eval_pics = np.array([x[-8:, 16:48] for x in x_eval_data[:, 3]]).reshape(-1, 256)

y_train_pics = np.array([np.rot90(x, -1)[-8:, 16:48] for x in y_train_data[:, 3]]).reshape(-1, 256)
y_eval_pics = np.array([np.rot90(x, -1)[-8:, 16:48] for x in y_eval_data[:, 3]]).reshape(-1, 256)

nx_train_pics = np.array([np.rot90(x, 2)[-8:, 16:48] for x in nx_train_data[:, 3]]).reshape(-1, 256)
nx_eval_pics = np.array([np.rot90(x, 2)[-8:, 16:48] for x in nx_eval_data[:, 3]]).reshape(-1, 256)

ny_train_pics = np.array([np.rot90(x, 1)[-8:, 16:48] for x in ny_train_data[:, 3]]).reshape(-1, 256)
ny_eval_pics = np.array([np.rot90(x, 1)[-8:, 16:48] for x in ny_eval_data[:, 3]]).reshape(-1, 256)

z_train_pics = np.array([x[24:40, 24:40] for x in z_train_data[:, 3]]).reshape(-1, 256)
z_eval_pics = np.array([x[24:40, 24:40] for x in z_eval_data[:, 3]]).reshape(-1, 256)

# test data
x_eval_pics = np.append(x_eval_pics, y_eval_pics).reshape(-1, 256)
x_eval_pics = np.append(x_eval_pics, nx_eval_pics).reshape(-1, 256)
x_eval_pics = np.append(x_eval_pics, ny_eval_pics).reshape(-1, 256)
x_eval_pics = np.append(x_eval_pics, z_eval_pics).reshape(-1, 256)

x_eval_pics = np.append(x_eval_pics, y_train_pics).reshape(-1, 256)
x_eval_pics = np.append(x_eval_pics, nx_train_pics).reshape(-1, 256)
x_eval_pics = np.append(x_eval_pics, ny_train_pics).reshape(-1, 256)
x_eval_pics = np.append(x_eval_pics, z_train_pics).reshape(-1, 256)

x_eval_labels = np.append(x_eval_labels, y_eval_labels)
x_eval_labels = np.append(x_eval_labels, nx_eval_labels)
x_eval_labels = np.append(x_eval_labels, ny_eval_labels)
x_eval_labels = np.append(x_eval_labels, z_eval_labels)

x_eval_labels = np.append(x_eval_labels, y_train_labels)
x_eval_labels = np.append(x_eval_labels, nx_train_labels)
x_eval_labels = np.append(x_eval_labels, ny_train_labels)
x_eval_labels = np.append(x_eval_labels, z_train_labels)

## train data and test data
scaler = StandardScaler()
X_test = scaler.fit_transform(x_eval_pics)

test_data = TestData(torch.FloatTensor(X_test))

test_loader = DataLoader(dataset=test_data, batch_size=1)

##################################################### Build Model #####################################################


# flatten the tensor into
class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


# sequential based model
seq_model = nn.Sequential(
           nn.Conv2d(1, 10, kernel_size=(3, 5)),
           nn.MaxPool2d((1, 2)),
           nn.ReLU(),
           nn.Dropout2d(),
           nn.Conv2d(10, 20, kernel_size=(3, 3)),
           nn.MaxPool2d((2, 3)),
           nn.ReLU(),
           Flatten(),
           nn.Linear(160, 80),
           nn.ReLU(),
           nn.Linear(80, 1)
         )

net = seq_model
print(net)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###################################################### Load Model ######################################################

net.load_state_dict(torch.load("CNN_Model_8.pk"))

#################################################### Evaluate Model ####################################################

y_pred_list = []
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = net(torch.reshape(X_batch, (1, 1, 8, 32)))
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
confusion_matrix(x_eval_labels, y_pred_list)
print(classification_report(x_eval_labels, y_pred_list))

