import numpy as np
from deep_heuristic.nn_utils import split_data, load_workspace
import torch
import torch.nn as nn
import pickle as pk

train_data, train_labels, eval_data, eval_labels = split_data(load_workspace(), test_size=0.2, num_of_param=3)

##################################################### Build Model #####################################################


# Binary classification
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred


model = NeuralNet1(input_size=3 * 1, hidden_size=10)
criterion = nn.BCELoss()

# 2) Loss and optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 3) Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(torch.tensor(train_data, dtype=torch.float32)).squeeze()
    loss = criterion(y_predicted, torch.tensor(train_labels, dtype=torch.float32))
    if (epoch + 1) % (num_epochs/10) == 0:
        print(y_predicted[:2], train_labels[:2])
    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    if (epoch + 1) % (num_epochs/10) == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

predicted = model(torch.tensor(eval_data, dtype=torch.float32)).detach().numpy()
n_correct = 0
n_samples = 0
for (i, ii) in zip(predicted, eval_labels):
    n_samples += 1
    if (i[0] >= 0.5 and ii) or (i[0] < 0.5 and not ii):
        n_correct += 1
acc = 100.0 * n_correct / n_samples
print(f'Accuracy of the network on the test images: {acc} %')
