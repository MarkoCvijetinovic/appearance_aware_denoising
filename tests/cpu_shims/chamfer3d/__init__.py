import torch


class ChamferDistance(torch.nn.Module):
    def forward(self, p1, p2):
        d = torch.cdist(p1, p2) ** 2
        dist1, idx1 = d.min(dim=2)
        dist2, idx2 = d.min(dim=1)
        return dist1, dist2, idx1.int(), idx2.int()
