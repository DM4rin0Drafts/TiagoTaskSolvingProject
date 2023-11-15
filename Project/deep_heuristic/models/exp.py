import numpy as np
import torch
import matplotlib.pyplot as plt
import load_model
from deep_heuristic.nn_utils import split_data, load_collision

x_dataset = np.array([])
for item in load_collision():
    item = np.array(item)
    tmp = np.array(item[0])
    if tmp[3] < 0.14:
        continue
    if tmp[0] == 0:
        x_dataset = np.append(x_dataset, (tmp[1:5], item[1]))
x_dataset = x_dataset.reshape(-1, 2)
x_train_data, x_train_labels, x_eval_data, x_eval_labels = split_data(x_dataset, test_size=0.01, num_of_param=4)

print(load_model.predict_grasp_direction(x_train_data[:10, 3]))

for i in range(0):
    arr = x_train_data[i, 3]
    print(x_train_labels[i])
    plt.imshow(arr, 'gray', vmin=0, vmax=1)
    plt.show()
