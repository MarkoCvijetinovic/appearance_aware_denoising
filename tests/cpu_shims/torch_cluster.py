import math

import torch


def fps(x, batch=None, ratio=0.5, random_start=True):
    n = x.shape[0]
    m = max(1, math.ceil(n * ratio))
    chosen = [int(torch.randint(n, (1,))) if random_start else 0]
    dist = torch.full((n,), float("inf"), device=x.device)
    for _ in range(m - 1):
        dist = torch.minimum(dist, ((x - x[chosen[-1]]) ** 2).sum(-1))
        chosen.append(int(torch.argmax(dist)))
    return torch.tensor(chosen, device=x.device)
