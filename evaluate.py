import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

seq_name = "sequence1"

database_idx_dic = {"sequence1": [[0, 10000]]}
test_query_idx_dic = {"sequence1": [[10000, 20000]]}

radar_odometry_file = open("../{}/heatmap data/odometry.txt".format(seq_name))

data_len = len(radar_odometry_file.readlines())
radar_odometry_file.close()
print("Data len:", data_len)

database_idx = []
test_query_idx = []
database_idx_list = database_idx_dic[seq_name]
test_query_idx_list = test_query_idx_dic[seq_name]
for idx in database_idx_list:
    database_idx += range(idx[0], idx[1])
for idx in test_query_idx_list:
    test_query_idx += range(idx[0], idx[1])

dbImageFns = []
qImageFns_test = []
utmDb = []
utmQ_test = []

file_format = '.npy'
radar_odometry_file = open("../{}/heatmap data/odometry.txt".format(seq_name))
for index, radar_odometry in enumerate(radar_odometry_file):
    radar_odometry = radar_odometry.split(" ")
    if index in database_idx:
        dbImageFns.append(radar_odometry[0] + file_format)
        utmDb.append([float(radar_odometry[2]), float(radar_odometry[3]), float(radar_odometry[4])])
    elif index in test_query_idx:
        qImageFns_test.append(radar_odometry[0] + file_format)
        utmQ_test.append([float(radar_odometry[2]), float(radar_odometry[3]), float(radar_odometry[4])])
radar_odometry_file.close()

posDistThr = 18
posDistSqThr = posDistThr**2
nonTrivPosDistSqThr = 3**2

ws = "test"
numImages = len(dbImageFns)
numQueries = len(qImageFns_test)
utmDb = np.array(utmDb)
utmQ_test = np.array(utmQ_test)
dbStruct = {
    "whichSet": ws,
    "dbImageFns": dbImageFns,
    "utmDb": utmDb.T,
    "qImageFns": qImageFns_test,
    "utmQ": utmQ_test.T,
    "numImages": numImages,
    "numQueries": numQueries,
    "posDistThr": posDistThr,
    "posDistSqThr": posDistSqThr,
    "nonTrivPosDistSqThr": nonTrivPosDistSqThr
}

fig = plt.figure(num=1, figsize=(16, 8))
ax = fig.add_subplot(111)
ax.scatter(utmDb[:, 0], utmDb[:, 1], s=0.1, c='r', label="Database")
ax.scatter(utmQ_test[:, 0], utmQ_test[:, 1], s=0.1, label="Query")
ax.set_xlabel('x(m)')
ax.set_ylabel('y(m)')
ax.set_title("database and query odometry")
ax.legend()
plt.show()

knn = NearestNeighbors(n_jobs=-1)
knn.fit(utmDb[:, :2])
_, gts = knn.radius_neighbors(utmQ_test[:, :2], radius=dbStruct["nonTrivPosDistSqThr"]**0.5, sort_results=True)

predictions = []  # TODO:
distances = []  # TODO:

# ------------------------------------------- recall@N ------------------------------------------- #
n_values = [1, 5, 10, 20, 50, 100]
correct_at_n = np.zeros(len(n_values))
for qIx, pred in enumerate(tqdm(predictions, desc='Get recall', ncols=50)):
    for i, n in enumerate(n_values):
        if np.any(np.in1d(pred[:n], gts[qIx])):
            correct_at_n[i:] += 1
            break
recall_at_n = correct_at_n / len(predictions) * 100.0
recalls = {}
for i, n in enumerate(n_values):
    recalls[n] = recall_at_n[i]
print('recall@1: {:.2f}%, recall@5: {:.2f}%, recall@10: {:.2f}%, recall@20: {:.2f}%,  recall@50: {:.2f}%, recall@100: {:.2f}%'.format(
    recalls[1], recalls[5], recalls[10], recalls[20], recalls[50], recalls[100]))

# ------------------------------------------- TP, FP, FN, TN------------------------------------------- #
distances = np.array(distances)
predictions = np.array(predictions)

dists_m = np.around(distances[:, 0], 2)
dists_u = np.array(list(set(dists_m)))
dists_u = np.sort(dists_u)

recalls_list = []
precisions_list = []
for th in tqdm(dists_u, ncols=40):
    TPCount = 0  # 预测为正类的正样本
    FPCount = 0  # 预测为负类的负样本
    FNCount = 0  # 预测为正类的负样本
    TNCount = 0  # 预测为负类的正样本
    for index_q in range(distances.shape[0]):
        if distances[index_q, 0] < th:
            if np.any(np.in1d(predictions[index_q, 0], gts[index_q])):
                TPCount += 1
            else:
                FPCount += 1
        else:
            if np.any(np.in1d(predictions[index_q, 0], gts[index_q])):
                FNCount += 1
            else:
                TNCount += 1
    assert TPCount + FPCount + FNCount + TNCount == dists_m.shape[0], 'Count Error!'
    if TPCount + FNCount == 0 or TPCount + FPCount == 0:
        continue
    recall = TPCount / (TPCount + FNCount)
    precision = TPCount / (TPCount + FPCount)
    recalls_list.append(recall)
    precisions_list.append(precision)

# ------------------------------------------- F1 Score ------------------------------------------- #
recalls = np.array(recalls_list)
precisions = np.array(precisions_list)
ind = np.argsort(recalls)
recalls = recalls[ind]
precisions = precisions[ind]
f1s = []
for index_j in range(len(recalls)):
    f1 = 2 * precisions[index_j] * recalls[index_j] / (precisions[index_j] + recalls[index_j])
    f1s.append(f1)
print('F1 Score:', max(f1s))

# --------------------------------------- Average Precision -------------------------------------- #
recalls = np.array(recalls_list)
precisions = np.array(precisions_list)
ind = np.argsort(recalls)
recalls = recalls[ind]
precisions = precisions[ind]
ap = 0
for index_j in range(len(recalls) - 1):
    ap += precisions[index_j] * (recalls[index_j + 1] - recalls[index_j])
print('Average Precision:', ap)
