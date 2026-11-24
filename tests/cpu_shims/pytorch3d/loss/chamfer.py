import torch


def chamfer_distance(x, y, batch_reduction="mean", point_reduction="mean", **kwargs):
    d = torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist") ** 2
    loss = d.min(dim=2)[0].mean(dim=1) + d.min(dim=1)[0].mean(dim=1)
    if batch_reduction == "mean":
        loss = loss.mean()
    return loss, None
