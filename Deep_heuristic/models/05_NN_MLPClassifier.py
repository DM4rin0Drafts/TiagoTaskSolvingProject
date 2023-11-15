import numpy as np
from deep_heuristic.nn_utils import split_data, load_workspace
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import pickle as pk
# import pandas as pd
import matplotlib.pyplot as plt

train_data, train_labels, eval_data, eval_labels = split_data(load_workspace(), test_size=0.1, num_of_param=3)
# 1 layer: 600:97.37 / 10:95.87 / 30:97.55 / 50:97.65 / 100:97.37 / 1000:97.35
# 2 layers: 10:96.66 / 100:96.86 / 500:97.88 / 1000:97.92
# 3 layers: 10:96.41 / 100:97.32 / 500:96.41 / 30.60.20:97.12
acc_list = np.array([])
for nodes in range(5, 51, 5):
    # clf_lbfgs = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(nodes, nodes), random_state=1)
    clf_adam = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=nodes, random_state=1)
    # clf_lbfgs.fit(train_data, train_labels+0)
    clf_adam.fit(train_data, train_labels+0)
    # cm_lbfgs = confusion_matrix(eval_labels, clf_lbfgs.predict(eval_data))
    cm_adam = confusion_matrix(eval_labels, clf_adam.predict(eval_data))
    acc_list = np.append(acc_list, [nodes, 1-(cm_adam[0, 1]+cm_adam[1, 0])/5070])
    # print(nodes, "\n lbfgs: ", 1-(cm_lbfgs[0, 1]+cm_lbfgs[1, 0])/5070, "\n")
    # print(nodes, "adam: ", 1-(cm_adam[0, 1]+cm_adam[1, 0])/5070, "\n")
    # plot_confusion_matrix(clf, eval_data, eval_labels+0)
    # plt.show()
for x in acc_list:
    print(x)
