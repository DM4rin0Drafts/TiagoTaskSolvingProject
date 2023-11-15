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
x_train_data, x_train_labels, x_eval_data, x_eval_labels = split_data(x_dataset, test_size=0., num_of_param=4)
y_train_data, y_train_labels, y_eval_data, y_eval_labels = split_data(y_dataset, test_size=0., num_of_param=4)
z_train_data, z_train_labels, z_eval_data, z_eval_labels = split_data(z_dataset, test_size=0., num_of_param=4)
nx_train_data, nx_train_labels, nx_eval_data, nx_eval_labels = split_data(nx_dataset, test_size=0., num_of_param=4)
ny_train_data, ny_train_labels, ny_eval_data, ny_eval_labels = split_data(ny_dataset, test_size=0., num_of_param=4)

"""
# load images
x_train_pics = np.array([x[-8:, 16:48] for x in x_train_data[:, 3]])
x_eval_pics = np.array([x[-8:, 20:40] for x in x_eval_data[:, 3]]).reshape(-1, 160)
y_train_pics = [x[16:48, -8:] for x in y_train_data[:, 3]]
z_train_pics = [x[16:48, 16:48] for x in z_train_data[:, 3]]
nx_train_pics = np.array([x[:16, 16:48] for x in nx_train_data[:, 3]]).reshape(-1, 512)
nx_eval_pics = np.array([x[:16, 16:48] for x in nx_eval_data[:, 3]]).reshape(-1, 512)
ny_train_pics = [x[16:48, :16] for x in ny_train_data[:, 3]]
"""

##################################################### Build Model #####################################################


def print_image(x):
    plt.imshow(x, 'gray', vmin=0, vmax=1)
    plt.show()


def calculate(train_data, train_labels, rot_idx=0, n_samples=-1, visu=False, debug=False):

    if n_samples < 0:
        n_samples = len(train_data)
    print("# of samples : ", n_samples)

    n_correct = 0
    f_p = 0
    f_n = 0
    ymean = np.array([])
    nymean = np.array([])
    ymin = np.array([])
    nymin = np.array([])

    for i, (x, y) in enumerate(zip(train_data[:n_samples, 3], train_labels[:n_samples])):
        x = np.rot90(x, rot_idx)
        if y:
            ymean = np.append(ymean, np.mean(x[-8:, 16:48]))
            ymin = np.append(ymin, np.min(x[-8:, 16:48]))
        else:
            nymean = np.append(nymean, np.mean(x[-8:, 16:48]))
            nymin = np.append(nymin, np.min(x[-8:, 16:48]))

        if (np.mean(x[-8:, 16:48]) > 0.85) == y:
            n_correct += 1
        elif y:
            if debug:
                print('fp: ', np.mean(x[-8:, 16:48]), np.min(x[-8:, 16:48]), np.max(x[-8:, 16:48]))
                print_image(x)
            f_p += 1
        else:
            if debug:
                print('fn: ', np.mean(x[-8:, 16:48]), np.min(x[-8:, 16:48]), np.max(x[-8:, 16:48]))
                print_image(x)
            f_n += 1

    acc = 100.0 * n_correct / n_samples
    f_p = 100.0 * f_p / n_samples
    f_n = 100.0 * f_n / n_samples

    print(f'Accuracy of the network on the test images: {acc} %')
    print(f'False positive of the network on the test images: {f_p} %')
    print(f'False negative of the network on the test images: {f_n} %')

    if 1:
        print('ymin: ', ymin.max(), ymin.min(), ', ymean: ', ymean.max(), ymean.min())
        print('nymin: ', nymin.max(), nymin.min(), ', nymean: ', nymean.max(), nymean.min())

    if visu:
        ymean = np.round_(ymean, decimals=4)
        nymean = np.round_(nymean, decimals=4)
        ymin = np.round_(ymin, decimals=4)
        nymin = np.round_(nymin, decimals=4)

        nymean.sort()
        ymean.sort()
        ymin.sort()
        nymin.sort()

        plt.plot(ymin, )
        plt.show()
        plt.plot(nymin, c='r')
        plt.show()
        plt.plot(ymean, )
        plt.show()
        plt.plot(nymean, c='r')
        plt.show()


calculate(x_train_data, x_train_labels)
calculate(y_train_data, y_train_labels, rot_idx=-1)
calculate(nx_train_data, nx_train_labels, rot_idx=2)
calculate(ny_train_data, ny_train_labels, rot_idx=1)
