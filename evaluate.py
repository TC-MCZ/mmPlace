import argparse

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WholeDatasetFromStruct
from train import encoder

parser = argparse.ArgumentParser()

parser.add_argument('--seq_name', type=str, default='sequence1')
parser.add_argument('--data_path', type=str, default='../sequence1/heatmap', help='Path for data.')
parser.add_argument('--optim', type=str, default='sgd', help='Optimizer to use', choices=['sgd', 'adam'])
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.000001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--margin', type=float, default=0.05, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restarts)')
parser.add_argument('--nEpochs', type=int, default=20, help='Number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=8, help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=128, help='Batch size for caching and testing')
parser.add_argument('--threads', type=int, default=1, help='Number of threads for each data loader to use')
parser.add_argument('--output_dim', type=int, default=6144, help='Number of feature dimension.')

args = parser.parse_args()


def gen_db(seq_name, vis=True):
    radar_odometry_path = "../" + seq_name + "/raw_data/radar_odometry.txt"
    radar_odometry_file = open(radar_odometry_path)
    data_len = len(radar_odometry_file.readlines())
    radar_odometry_file.close()

    database_idx = []
    test_query_idx = []
    database_idx_list = [[0, 10500]]
    test_query_idx_list = [[10500, 19000]]
    for idx in database_idx_list:
        database_idx += range(idx[0], idx[1])
    for idx in test_query_idx_list:
        test_query_idx += range(idx[0], idx[1])

    dbImageFns = []
    qImageFns_test = []
    utmDb = []
    utmQ_test = []

    radar_odometry_file = open(radar_odometry_path)
    for index, radar_odometry in enumerate(radar_odometry_file):
        radar_odometry = radar_odometry.split(" ")
        if index in database_idx:
            dbImageFns.append(radar_odometry[0] + ".npy")
            utmDb.append([float(radar_odometry[2]), float(radar_odometry[3]), float(radar_odometry[4])])
        elif index in test_query_idx:
            qImageFns_test.append(radar_odometry[0] + ".npy")
            utmQ_test.append([float(radar_odometry[2]), float(radar_odometry[3]), float(radar_odometry[4])])
    radar_odometry_file.close()

    posDistThr = 18
    posDistSqThr = posDistThr**2
    nonTrivPosDistSqThr = 3**2

    utmDb = np.array(utmDb)
    utmQ_test = np.array(utmQ_test)
    db_struct_test = {
        "whichSet": "test",
        "dbImageFns": dbImageFns,
        "utmDb": utmDb,
        "qImageFns": qImageFns_test,
        "utmQ": utmQ_test,
        "numImages": len(dbImageFns),
        "numQueries": len(utmQ_test),
        "posDistThr": posDistThr,
        "posDistSqThr": posDistSqThr,
        "nonTrivPosDistSqThr": nonTrivPosDistSqThr
    }

    if vis:
        plt.rcParams['figure.figsize'] = [16, 8]
        plt.rcParams['axes.titlesize'] = 20
        fig, axs = plt.subplots(1, 2)
        plt.subplots_adjust(hspace=0.5)

        axs[0].scatter(utmDb[:, 0], utmDb[:, 1])
        axs[0].set_title('DataBase')
        axs[0].tick_params(labelsize=20)
        axs[0].set_xlabel('X(m)', fontsize=20)
        axs[0].set_ylabel('Y(m)', fontsize=20)

        axs[1].scatter(utmQ_test[:, 0], utmQ_test[:, 1])
        axs[1].set_title('Test Query')
        axs[1].tick_params(labelsize=20)
        axs[1].set_xlabel('X(m)', fontsize=20)
        axs[1].set_ylabel('Y(m)', fontsize=20)

        plt.show()

    return db_struct_test


def evaluate(args, db_struct_test):
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = encoder()

    checkpoint = torch.load("checkpoint_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(device)
    model = model.to(device)

    test_set = WholeDatasetFromStruct(db_struct_test, args.data_path)
    test_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.cacheBatchSize, shuffle=False)

    model.eval()
    with torch.no_grad():
        dbFeat = np.empty((len(test_set), args.output_dim))
        for input, indices, _, _ in tqdm(test_data_loader, ncols=40):
            input = input.to(device)
            encoder_feature = model(input)
            dbFeat[indices.detach().numpy(), :] = encoder_feature.detach().cpu().numpy()
            del input, indices
    del test_data_loader

    n_values = [1, 5, 10, 20]
    qFeat = dbFeat[test_set.db_struct["numImages"]:].astype('float32')
    dbFeat = dbFeat[:test_set.db_struct["numImages"]].astype('float32')

    faiss_index = faiss.IndexFlatL2(args.output_dim)
    faiss_index.add(dbFeat)
    _, predictions = faiss_index.search(qFeat, len(dbFeat))

    _, gt = test_set.getPositives()

    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / test_set.db_struct["numQueries"] * 100.0

    recalls = {}
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
    print('recall@1: {:.2f}%, recall@5: {:.2f}%, recall@10: {:.2f}%, recall@20: {:.2f}%'.format(recalls[1], recalls[5], recalls[10], recalls[20]))


if __name__ == '__main__':
    db_struct_test = gen_db(args.seq_name)

    evaluate(args, db_struct_test)

