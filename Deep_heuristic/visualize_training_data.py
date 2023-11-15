from utils.pybullet_tools.utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from deep_heuristic.nn_utils import split_data

file_reach = '../training_data/reach.pk'
# all_data = read_pickle(file_reach)
with open(file_reach, 'rb') as f:
    all_data = pickle.load(f)

train_data, train_labels, eval_data, eval_labels = split_data(all_data, test_size=0.)


train_data_w_true = []
train_data_x_true = []
train_data_y_true = []
train_data_w_false = []
train_data_x_false = []
train_data_y_false = []

for d, l in zip(train_data, train_labels):
    if l:
        train_data_w_true.append(d[0])
        train_data_x_true.append(d[1])
        train_data_y_true.append(d[2])
    else:
        train_data_w_false.append(d[0])
        train_data_x_false.append(d[1])
        train_data_y_false.append(d[2])

plt.scatter(train_data_x_true, train_data_w_true, color='red', s=np.array(train_data_y_true) * 5, alpha=0.4,
            edgecolors='w')
plt.scatter(train_data_x_false, train_data_w_false, color='blue', s=np.array(train_data_y_false) * 5, alpha=0.4,
            edgecolors='w')
ax = plt.gca()
ax.scatter(clf.support_vectors_[:, 1], clf.support_vectors_[:, 0], facecolors='none', edgecolors='g')
plt.xlabel('x')
plt.ylabel('Distance')
# ax.plot(np.degrees(train_data_x_true), train_data_w_true, '.', color='red')
# ax.plot(np.degrees(train_data_x_false), train_data_w_false, '.', color='blue')

########## show 3D data
fig = plt.figure(figsize=(8, 6))
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
# ax = fig.add_subplot(111, projection='3d')

ax.scatter(train_data_w_true, train_data_x_true, train_data_y_true, s=10, alpha=0.6, edgecolors='w', color='red')
ax.scatter(train_data_w_false, train_data_x_false, train_data_y_false, s=10, alpha=0.6, edgecolors='w', color='blue')

ax.set_xlabel('distance')
ax.set_ylabel('dir_jj')
ax.set_zlabel('z_jj')
# relation between the distance and the angle between the grasping direction
# and the line connecting surface center with robot base
plt.title('relation between the distance and the angle dir_jj', y=1.05)

########## save images
for ii in range(0, 360, 30):
        ax.view_init(elev=10., azim=ii)
        plt.savefig("movie%d.png" % ii)
        
plt.show()
