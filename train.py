import argparse
import os

import faiss
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import QueryDatasetFromStruct, WholeDatasetFromStruct, collate_fn

parser = argparse.ArgumentParser()

parser.add_argument('--seq_name', type=str, default='sequence0')
parser.add_argument('--data_path', type=str, default='../sequence0/heatmap', help='Path for data.')
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


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4), padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4), padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4), padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4), padding=0, dilation=1, ceil_mode=False),
        )

    def forward(self, input):
        input = input.type(torch.cuda.FloatTensor)

        batch_size, input_c, input_h, input_w = input.shape

        feature = self.encoder(input)
        feature = feature.view(batch_size, -1)

        feature = F.normalize(feature, p=2, dim=1)

        return feature


def gen_db(seq_name, vis=True):
    radar_odometry_path = "../" + seq_name + "/raw_data/radar_odometry.txt"
    radar_odometry_file = open(radar_odometry_path)
    data_len = len(radar_odometry_file.readlines())
    radar_odometry_file.close()

    database_len = 5800
    train_query_len = 5800
    val_query_len = 5800
    test_query_len = data_len - database_len - train_query_len - val_query_len
    print("Database len: {}, Train query: {}, Val query: {}, Test quey: {}".format(database_len, train_query_len, val_query_len, test_query_len))

    utmDb = np.empty((database_len, 3))
    utmQ_train = np.empty((train_query_len, 3))
    utmQ_val = np.empty((val_query_len, 3))
    utmQ_test = np.empty((test_query_len, 3))

    dbImageFns = []
    qImageFns_train = []
    qImageFns_val = []
    qImageFns_test = []

    radar_odometry_file = open(radar_odometry_path)
    for index, radar_odometry in enumerate(radar_odometry_file):
        radar_odometry = radar_odometry.split(" ")
        if index < database_len:
            dbImageFns.append(radar_odometry[0] + ".npy")
            utmDb[index][0] = float(radar_odometry[2])
            utmDb[index][1] = float(radar_odometry[3])
            utmDb[index][2] = float(radar_odometry[4])
        elif index < database_len + train_query_len:
            qImageFns_train.append(radar_odometry[0] + ".npy")
            utmQ_train[index - database_len][0] = float(radar_odometry[2])
            utmQ_train[index - database_len][1] = float(radar_odometry[3])
            utmQ_train[index - database_len][2] = float(radar_odometry[4])
        elif index < database_len + train_query_len + val_query_len:
            qImageFns_val.append(radar_odometry[0] + ".npy")
            utmQ_val[index - train_query_len - database_len][0] = float(radar_odometry[2])
            utmQ_val[index - train_query_len - database_len][1] = float(radar_odometry[3])
            utmQ_val[index - train_query_len - database_len][2] = float(radar_odometry[4])
        elif index < database_len + train_query_len + val_query_len + test_query_len:
            qImageFns_test.append(radar_odometry[0] + ".npy")
            utmQ_test[index - val_query_len - train_query_len - database_len][0] = float(radar_odometry[2])
            utmQ_test[index - val_query_len - train_query_len - database_len][1] = float(radar_odometry[3])
            utmQ_test[index - val_query_len - train_query_len - database_len][2] = float(radar_odometry[4])
    radar_odometry_file.close()

    posDistThr = 18
    posDistSqThr = posDistThr**2
    nonTrivPosDistSqThr = 3**2

    db_struct_train = {
        "whichSet": "train",
        "dbImageFns": dbImageFns,
        "utmDb": utmDb,
        "qImageFns": qImageFns_train,
        "utmQ": utmQ_train,
        "numImages": len(dbImageFns),
        "numQueries": len(qImageFns_train),
        "posDistThr": posDistThr,
        "posDistSqThr": posDistSqThr,
        "nonTrivPosDistSqThr": nonTrivPosDistSqThr
    }
    db_struct_val = {
        "whichSet": "val",
        "dbImageFns": dbImageFns,
        "utmDb": utmDb,
        "qImageFns": qImageFns_val,
        "utmQ": utmQ_val,
        "numImages": len(dbImageFns),
        "numQueries": len(utmQ_val),
        "posDistThr": posDistThr,
        "posDistSqThr": posDistSqThr,
        "nonTrivPosDistSqThr": nonTrivPosDistSqThr
    }
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
        fig, axs = plt.subplots(2, 2)
        plt.subplots_adjust(hspace=0.5)

        axs[0][0].scatter(utmDb[:, 0], utmDb[:, 1])
        axs[0][0].set_title('DataBase')
        axs[0][0].tick_params(labelsize=20)
        axs[0][0].set_xlabel('X(m)', fontsize=20)
        axs[0][0].set_ylabel('Y(m)', fontsize=20)

        axs[0][1].scatter(utmQ_train[:, 0], utmQ_train[:, 1])
        axs[0][1].set_title('Query_train')
        axs[0][1].tick_params(labelsize=20)
        axs[0][1].set_xlabel('X(m)', fontsize=20)
        axs[0][1].set_ylabel('Y(m)', fontsize=20)

        axs[1][0].scatter(utmQ_val[:, 0], utmQ_val[:, 1])
        axs[1][0].set_title('Query_val')
        axs[1][0].tick_params(labelsize=20)
        axs[1][0].set_xlabel('X(m)', fontsize=20)
        axs[1][0].set_ylabel('Y(m)', fontsize=20)

        axs[1][1].scatter(utmQ_test[:, 0], utmQ_test[:, 1])
        axs[1][1].set_title('Query_test')
        axs[1][1].tick_params(labelsize=20)
        axs[1][1].set_xlabel('X(m)', fontsize=20)
        axs[1][1].set_ylabel('Y(m)', fontsize=20)

        plt.show()

    return db_struct_train, db_struct_val, db_struct_test


def get_recall(args, model, val_set, device):
    test_data_loader = DataLoader(dataset=val_set, num_workers=args.threads, batch_size=args.cacheBatchSize, shuffle=False)

    print('----------------------------eval..----------------------------')
    model.eval()
    with torch.no_grad():
        dbFeat = np.empty((len(val_set), args.output_dim))
        for input, indices, _, _ in tqdm(test_data_loader, ncols=40):
            input = input.to(device)
            encoder_feature = model(input)
            dbFeat[indices.detach().numpy(), :] = encoder_feature.detach().cpu().numpy()
            del input, indices
    del test_data_loader

    n_values = [1, 5, 10, 20]
    qFeat = dbFeat[val_set.db_struct["numImages"]:].astype('float32')
    dbFeat = dbFeat[:val_set.db_struct["numImages"]].astype('float32')

    faiss_index = faiss.IndexFlatL2(args.output_dim)
    faiss_index.add(dbFeat)
    _, predictions = faiss_index.search(qFeat, len(dbFeat))

    _, gt = val_set.getPositives()

    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / val_set.db_struct["numQueries"] * 100.0

    recalls = {}
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]

    return recalls


def train(args, db_struct_train, db_struct_val):
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = encoder()
    model = model.to(device)

    if args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weightDecay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lrStep, gamma=args.lrGamma)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)

    feature_criterion = nn.TripletMarginLoss(margin=args.margin**0.5, p=2, reduction='sum').to(device)

    whole_train_set = WholeDatasetFromStruct(db_struct_train, args.data_path)
    train_set = QueryDatasetFromStruct(db_struct_train, args.data_path)
    val_set = WholeDatasetFromStruct(db_struct_val, args.data_path)

    whole_train_data_loader = DataLoader(dataset=whole_train_set, num_workers=args.threads, batch_size=args.cacheBatchSize, shuffle=False)

    best_recall_at_1 = 0
    for epoch in range(args.start_epoch + 1, args.nEpochs + 1):
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        print('----------------------------build cache..----------------------------')
        model.eval()
        train_set.cache = os.path.join(train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5:
            h5feat = h5.create_dataset('features', [len(whole_train_set), args.output_dim], dtype=np.float32)
            with torch.no_grad():
                for input, indices, _, _ in tqdm(whole_train_data_loader, ncols=100):
                    input = input.to(device)
                    encoder_feature = model(input)
                    h5feat[indices.detach().numpy(), :] = encoder_feature.detach().cpu().numpy()
                    del input, encoder_feature

        epoch_loss = 0
        startIter = 1
        nBatches = (len(train_set) + args.batchSize - 1) // args.batchSize

        print('----------------------------training..----------------------------')
        model.train()
        train_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True, collate_fn=collate_fn)
        for iteration, (query, positives, negatives, negCounts, indices, _) in enumerate(tqdm(train_data_loader, ncols=100), startIter):
            if query is None:
                continue

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            input = torch.cat([query, positives, negatives])

            input = input.to(device)
            encoder_feature = model(input)

            q_feature, p_feature, n_feature = torch.split(encoder_feature, [B, B, nNeg])

            optimizer.zero_grad()

            feature_loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    feature_loss += feature_criterion(q_feature[i:i + 1], p_feature[i:i + 1], n_feature[negIx:negIx + 1])
            feature_loss /= nNeg.float().to(device)

            feature_loss.backward()
            optimizer.step()

            epoch_loss += feature_loss.item()

            del input, encoder_feature, q_feature, p_feature, n_feature
            del query, positives, negatives
            del feature_loss
        startIter += len(train_data_loader)
        train_loss = epoch_loss / nBatches
        os.remove(train_set.cache)

        if args.optim == 'sgd':
            scheduler.step()

        current_recalls = get_recall(args, model, val_set, device)
        is_best = current_recalls[1] > best_recall_at_1
        if is_best:
            best_recall_at_1 = current_recalls[1]
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'recalls': current_recalls,
                'best_recall_at_1': best_recall_at_1,
                'optimizer': optimizer.state_dict()
            }, "checkpoint_best.pth.tar")
        print(
            'epoch: {:>2d}\t'.format(epoch),
            'lr: {:>.8f}\t'.format(current_lr),
            'train loss: {:>.4f}\t'.format(train_loss),
            'recall@1: {:.2f}\t'.format(current_recalls[1]),
            'recall@5: {:.2f}\t'.format(current_recalls[5]),
            'recall@10: {:.2f}\t'.format(current_recalls[10]),
            'recall@20: {:.2f}\t'.format(current_recalls[20]),
            '**best**\n' if is_best else '\n',
        )


if __name__ == '__main__':
    db_struct_train, db_struct_val, _ = gen_db(args.seq_name)

    train(args, db_struct_train, db_struct_val)

