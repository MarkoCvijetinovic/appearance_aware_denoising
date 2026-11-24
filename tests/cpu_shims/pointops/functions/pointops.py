import torch


def _segments(offset):
    start = 0
    for end in offset.tolist():
        yield start, int(end)
        start = int(end)


def furthestsampling(xyz, offset, new_offset):
    idx = []
    for (s, e), (ns, ne) in zip(_segments(offset), _segments(new_offset)):
        pts, m = xyz[s:e], ne - ns
        chosen = [0]
        dist = torch.full((e - s,), float("inf"), device=xyz.device)
        for _ in range(m - 1):
            dist = torch.minimum(dist, ((pts - pts[chosen[-1]]) ** 2).sum(-1))
            chosen.append(int(torch.argmax(dist)))
        idx.append(torch.tensor(chosen, device=xyz.device) + s)
    return torch.cat(idx).int()


def knnquery(nsample, xyz, new_xyz, offset, new_offset):
    if new_xyz is None:
        new_xyz = xyz
    idx, dist = [], []
    for (s, e), (ns, ne) in zip(_segments(offset), _segments(new_offset)):
        d = torch.cdist(new_xyz[ns:ne], xyz[s:e])
        dd, ii = d.topk(nsample, dim=1, largest=False)
        idx.append(ii + s)
        dist.append(dd)
    return torch.cat(idx).int(), torch.cat(dist)


def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True, return_index=False):
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset)
    idx = idx.view(-1).long()
    m, c = new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx, :].view(m, nsample, 3) - new_xyz.unsqueeze(1)
    grouped_feat = feat[idx, :].view(m, nsample, c)
    if use_xyz:
        grouped_feat = torch.cat((grouped_xyz, grouped_feat), -1)
    if return_index:
        return grouped_feat, idx
    return grouped_feat


def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset)
    dist_recip = 1.0 / (dist + 1e-8)
    weight = dist_recip / dist_recip.sum(dim=1, keepdim=True)
    new_feat = torch.zeros(new_xyz.shape[0], feat.shape[1], dtype=feat.dtype, device=feat.device)
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat
