import itertools
from os.path import join

import h5py
import numpy as np
import torch
import torch.utils.data as data
from sklearn.neighbors import NearestNeighbors


class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, db_struct, img_dir, onlyDB=False):
        super().__init__()
        self.db_struct = db_struct
        self.whichSet = db_struct["whichSet"]
        self.images = [join(img_dir, dbIm) for dbIm in self.db_struct["dbImageFns"]]
        self.utms = [utmDb for utmDb in self.db_struct["utmDb"]]
        if not onlyDB:
            self.images += [join(img_dir, qIm) for qIm in self.db_struct["qImageFns"]]
            self.utms += [utmQ for utmQ in self.db_struct["utmQ"]]
        self.positives = None
        self.distances = None

    def load_data(self, index):
        filename = self.images[index]
        frame_index = int(filename[-9:-4])
        img = np.load(filename[:-9] + '{:0>5d}.npy'.format(int(frame_index)))
        img = torch.from_numpy(img).unsqueeze(dim=0)
        img = img / img.max()

        return img, index, filename

    def __getitem__(self, index):
        imgs, index, filename = self.load_data(index)
        pos = self.utms[index].astype(np.float32)

        return imgs, index, filename, pos

    def getPositives(self):
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_struct["utmDb"][:, :2])
            self.distances, self.positives = knn.radius_neighbors(self.db_struct["utmQ"][:, :2], radius=self.db_struct["nonTrivPosDistSqThr"]**0.5)

        return self.distances, self.positives

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None, None, None

    query, positive, negatives, indices, pos_gt = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    pos_gt = data.dataloader.default_collate(pos_gt)
    indices = list(itertools.chain(*indices))

    if query is None or positive is None or negatives is None or negCounts is None or indices is None or pos_gt is None:
        print('exit none, query: {}, positive: {}, negatives: {}, neCounts: {}, indices: {}, pos_gt: {}'.format(query, positive, negatives, negCounts, indices, pos_gt))

    return query, positive, negatives, negCounts, indices, pos_gt


class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, db_struct, img_dir, nNegSample=1000, nNeg=10, margin=0.1):
        super().__init__()
        self.db_struct = db_struct
        self.whichSet = db_struct["whichSet"]
        self.img_dir = img_dir
        self.nNegSample = nNegSample
        self.nNeg = nNeg
        self.margin = margin

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_struct["utmDb"][:, :2])

        self.nontrivial_positives = list(knn.radius_neighbors(self.db_struct["utmQ"][:, :2], radius=self.db_struct["nonTrivPosDistSqThr"]**0.5, return_distance=False))
        for i, pos in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(pos)

        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives]) > 0)[0]

        potential_positives = knn.radius_neighbors(self.db_struct["utmQ"][:, :2], radius=self.db_struct["posDistThr"], return_distance=False)
        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.db_struct["numImages"]), pos, assume_unique=True))

        self.negCache = [np.empty((0, )) for _ in range(self.db_struct["numQueries"])]

        self.cache = None

    def load_data(self, filename):
        frame_index = int(filename[-9:-4])
        img = np.load(filename[:-9] + '{:0>5d}.npy'.format(int(frame_index)))
        img = torch.from_numpy(img).unsqueeze(dim=0)
        img = img / img.max()

        return img

    def __getitem__(self, index):
        index = self.queries[index]
        with h5py.File(self.cache, mode='r') as h5:
            h5feat = h5.get("features")
            qOffset = self.db_struct["numImages"]

            qFeat = h5feat[index + qOffset]
            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            qFeat = torch.tensor(qFeat)
            posFeat = torch.tensor(posFeat)
            dist = torch.norm(qFeat - posFeat, dim=1, p=None)
            result = dist.topk(1, largest=False)
            dPos, posNN = result.values, result.indices
            posIndex = self.nontrivial_positives[index][posNN].item()

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))

            negFeat = h5feat[negSample.tolist()]
            negFeat = torch.tensor(negFeat)
            dist = torch.norm(qFeat - negFeat, dim=1, p=None)
            result = dist.topk(self.nNeg * 10, largest=False)
            dNeg, negNN = result.values, result.indices

            violatingNeg = dNeg.numpy() < dPos.numpy() + self.margin**0.5

            if np.sum(violatingNeg) < 1:
                return None

            negNN = negNN.numpy()
            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        query = self.load_data(join(self.img_dir, self.db_struct["qImageFns"][index]))
        positive = self.load_data(join(self.img_dir, self.db_struct["dbImageFns"][posIndex]))
        query_pos = self.db_struct["utmQ"][index].astype(np.float32)
        positive_pos = self.db_struct["utmDb"][posIndex].astype(np.float32)
        pos = query_pos - positive_pos
        if query_pos[2] - positive_pos[2] > 180:
            pos[2] = query_pos[2] - positive_pos[2] - 360
        elif query_pos[2] - positive_pos[2] < -180:
            pos[2] = query_pos[2] - positive_pos[2] + 360

        negatives = []
        for negIndex in negIndices:
            negative = self.load_data(join(self.img_dir, self.db_struct["dbImageFns"][negIndex]))
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex] + negIndices.tolist(), pos

    def __len__(self):
        return len(self.queries)

