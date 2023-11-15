import numpy as np
from sklearn.svm import NuSVC
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from deep_heuristic.nn_utils import split_data, load_workspace
import pickle as pk
import matplotlib.pyplot as plt

train_data, train_labels, eval_data, eval_labels = split_data(load_workspace(), test_size=0.1, num_of_param=3)

clf = NuSVC(gamma='auto')
clf.fit(train_data, train_labels)
cm_adam = confusion_matrix(eval_labels, clf.predict(eval_data))
acc_list = 1 - (cm_adam[0, 1] + cm_adam[1, 0]) / np.sum(cm_adam)
print(acc_list)
plot_confusion_matrix(clf, eval_data, eval_labels)
plt.show()
