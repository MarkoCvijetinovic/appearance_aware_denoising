from collections import namedtuple

import torch

_KNN = namedtuple("KNN", ["dists", "idx", "knn"])


def knn_points(p1, p2, K=1, return_nn=False, **kwargs):
    d = torch.cdist(p1, p2) ** 2
    dists, idx = d.topk(K, dim=2, largest=False)
    nn = torch.stack([p2[b][idx[b]] for b in range(p2.shape[0])]) if return_nn else None
    return _KNN(dists, idx, nn)
