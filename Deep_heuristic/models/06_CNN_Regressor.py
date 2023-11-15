import matplotlib.pyplot as plt
import numpy as np
from deep_heuristic.nn_utils import split_data, load_collision, TrainData, TestData
import torch
import torch.nn as nn
import pickle as pk

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
x_train_pics = np.array([x[-8:, 20:40] for x in x_train_data[:, 3]]).reshape(-1, 160)
x_eval_pics = np.array([x[-8:, 20:40] for x in x_eval_data[:, 3]]).reshape(-1, 160)
y_train_pics = [x[16:48, -16:] for x in y_train_data[:, 3]]
z_train_pics = [x[16:48, 16:48] for x in z_train_data[:, 3]]
nx_train_pics = np.array([x[:16, 16:48] for x in nx_train_data[:, 3]]).reshape(-1, 512)
nx_eval_pics = np.array([x[:16, 16:48] for x in nx_eval_data[:, 3]]).reshape(-1, 512)
ny_train_pics = [x[16:48, :16] for x in ny_train_data[:, 3]]


##################################################### Build Model #####################################################

# Binary classification
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.linear3 = nn.Linear(int(hidden_size/2), 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred

input_size = 8 * 20 + 3
model = NeuralNet1(input_size=input_size, hidden_size=80)
criterion = nn.MSELoss()

# 2) Loss and optimizer
learning_rate = 0.1
optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)

# 3) Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass and loss
    # y_predicted = model(
    # torch.tensor(np.append(x_train_pics, np.flip(nx_train_pics)).reshape(-1, 512), dtype=torch.float32)).squeeze()
    y_predicted = model(
        torch.tensor(np.append(x_train_pics, np.array(x_train_data[:, :3], dtype=float), axis=1).reshape(-1, input_size), dtype=torch.float32)).squeeze()
    loss = criterion(y_predicted, torch.tensor(x_train_labels, dtype=torch.float32))

    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    if (epoch + 1) % (num_epochs/10) == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

# predicted = model(torch.tensor(x_eval_pics, dtype=torch.float32)).detach().numpy()
predicted = model(torch.tensor(np.append(x_eval_pics, np.array(x_eval_data[:, :3], dtype=float), axis=1).reshape(-1, input_size), dtype=torch.float32)).detach().numpy()
n_correct = 0
n_samples = 0
f_p = 0
f_n = 0
minfp = 1
index = 0
for (i, ii) in zip(predicted, x_eval_labels):
    n_samples += 1
    if i[0] >= 0.5:
        if ii:
            n_correct += 1
        else:
            f_n += 1
    elif not ii:
        n_correct += 1
    else:
        f_p += 1
        if i[0] < minfp:
            minfp = i[0]
        plt.imshow(np.array(x_eval_data[index, 3]).reshape(64, 64), 'gray', vmin=0, vmax=1)
        plt.show()
        plt.imshow(np.array(x_eval_pics[index]).reshape(8, 20), 'gray', vmin=0, vmax=1)
        plt.show()
    index += 1

acc = 100.0 * n_correct / n_samples
f_p = 100.0 * f_p / n_samples
f_n = 100.0 * f_n / n_samples

print(f'Accuracy of the network on the test images: {acc} %')
print(f'False negative of the network on the test images: {f_n} ')
print(f'False positive of the network on the test images: {f_p} ', minfp)
