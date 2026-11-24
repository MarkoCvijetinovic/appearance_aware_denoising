import types
from contextlib import contextmanager

import torch
from torch import nn

from src.pgd.loader import DEFAULT_CHECKPOINT, add_pgd_to_path, load_pgd_network

add_pgd_to_path()
from pointops.functions import pointops

VARIANTS = ("geometry_only", "grayscale", "rgb_fusion", "gated")
LAB_SCALE = 100.0 # durch 100 geteilt liegen die Farbunterschiede ca. zw 0 & 1, selbe Größenordnung wie die Koordinaten auf der Einheitskugel
N_NEIGHBOURS = 16
COLOUR_BLOCK = 1 # hängen Farbe nur in den 2 Encoder-Block ein, also Index 1, trainieren nur diesen


def convert_rgb_to_lab(rgb):
    # in Lab entsprechen gleiche Abstände ungefähr gleich stark wahrgenommenen Unterschieden 
    rgb = rgb.clamp(0.0, 1.0) # clamp stellt sicher, dass die Werte zwischen 0 und 1 liegen
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    to_xyz = rgb.new_tensor([[0.4124564, 0.3575761, 0.1804375],
                             [0.2126729, 0.7151522, 0.0721750],
                             [0.0193339, 0.1191920, 0.9503041]])
    xyz = linear @ to_xyz.T / rgb.new_tensor([0.95047, 1.0, 1.08883])
    eps = 216 / 24389
    f = torch.where(xyz > eps, xyz.clamp(min=eps) ** (1 / 3), (24389 / 27 * xyz + 16) / 116) # lab function
    lightness = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([lightness, a, b], dim=-1)


def compute_neighbour_colour_differences(lab, neighbour_idx, n_neighbours=N_NEIGHBOURS):
    #returned pro Nachbar zwei Zahlen:
        # den Betrag des Helligkeitsunterschieds, |ΔL|
        # die Länge des Unterschieds in der a-b-Ebene, ‖Δab‖
    neighbours = lab[neighbour_idx].view(-1, n_neighbours, 3)
    delta = (neighbours - lab.unsqueeze(1)) / LAB_SCALE
    return torch.stack([delta[..., 0].abs(), delta[..., 1:].norm(dim=-1)], dim=-1)


def compute_gate_inputs(relative_xyz, colour_differences):
    with torch.autocast(relative_xyz.device.type, enabled=False):
        xyz = relative_xyz.float()
        centred = xyz - xyz.mean(dim=1, keepdim=True)
        covariance = centred.transpose(1, 2) @ centred / xyz.shape[1]
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp(min=0)
    surface_variation = 3 * eigenvalues[:, 0] / eigenvalues.sum(dim=1).clamp(min=1e-12)
    return torch.cat([surface_variation.unsqueeze(1), colour_differences.float().mean(dim=1)], dim=1)


class ColourGate(nn.Module):
    def __init__(self, n_inputs=3, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs):
        return torch.sigmoid(self.net(inputs))


class ColourCueSelector(nn.Module):
    # hier der Kernunterschied zu PGD
    def __init__(self, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
        self.variant = variant
        self.gate = ColourGate()

    def forward(self, relative_xyz, cues):
        if self.variant == "geometry_only":
            return relative_xyz.new_zeros(*relative_xyz.shape[:2], 2)
        if self.variant == "grayscale":
            return cues * cues.new_tensor([1.0, 0.0])
        if self.variant == "gated":
            g = self.gate(compute_gate_inputs(relative_xyz, cues))
            return cues * g.unsqueeze(1).to(cues.dtype)
        return cues


class ColourRFE(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.colour_conv = nn.Conv1d(2, base.BiMLP[0].out_channels, 1, bias=False)
        nn.init.zeros_(self.colour_conv.weight) # makes all varianten to begin at same weights (from original pgd baseline) 

    def forward(self, p, x, colour):
        base = self.base
        centre = p[:, 0, :].unsqueeze(1).expand(-1, p.shape[1], -1)
        dist = torch.sqrt(torch.sum((centre - p) ** 2, dim=2, keepdim=True))
        geometry = torch.cat([centre, p, centre - p, dist], dim=-1)
        hidden = base.BiMLP[0](geometry.permute(0, 2, 1)) + self.colour_conv(colour.permute(0, 2, 1).to(geometry.dtype)) # unterschied zu original
        p_c = base.BiMLP[1:](hidden).permute(0, 2, 1).contiguous()
        p_x = torch.cat([p_c, x], dim=-1)
        scores = base.score_fn(p_x)
        features = torch.sum(scores * p_x, dim=1, keepdim=True)
        return base.mlp_out(features.squeeze())


class ColourMRE(nn.Module):
    def __init__(self, base, variant):
        super().__init__()
        self.base = base
        self.rfe_1 = ColourRFE(base.Rfe_1)
        self.selector = ColourCueSelector(variant)
        self.lab = None

    def select_colour_cues(self, relative_xyz, neighbour_idx):
        return self.selector(relative_xyz, compute_neighbour_colour_differences(self.lab, neighbour_idx))

    def forward(self, p, x, o):
        base = self.base
        x_start = x
        x = base.mlp0(x)
        xr, idx = pointops.queryandgroup(N_NEIGHBOURS, p, p, x, None, offset=o, new_offset=o, use_xyz=True,
                                         return_index=True)
        x = self.rfe_1(xr[:, :, :3], xr[:, :, 3:], self.select_colour_cues(xr[:, :, :3], idx))
        x_middle = x
        xr, _ = pointops.queryandgroup(N_NEIGHBOURS, p, p, x, None, offset=o, new_offset=o, use_xyz=True,
                                       return_index=True)
        x = base.Rfe_2(xr[:, :, :3], xr[:, :, 3:])
        x = torch.cat([x_middle, x], dim=1)
        return base.mlp01(x_start) + base.mlp1(x)

#### Codebook Korrektur ####
def quantize_with_exact_gradient(self, features, calculate_commitment_loss=False):
    quantized, _, _ = self.soft_quantize(features)
    return features + (quantized - features), features.new_zeros(())


def enable_exact_codebook_gradients(network):
    for codebook in network.codebooks:
        if codebook is not None:
            codebook.forward = types.MethodType(quantize_with_exact_gradient, codebook)
#############################

class ColourWeightedPGD(nn.Module):
    # das Gesamtmodell
    def __init__(self, network, variant="gated"):
        super().__init__()
        self.network = network
        enable_exact_codebook_gradients(network)
        block = network.encoder_blocks[COLOUR_BLOCK]
        block.mre = ColourMRE(block.mre, variant)
        self.network.requires_grad_(False)
        for module in self.get_trainable_roots():
            module.requires_grad_(True)

    @property
    def colour_mre(self):
        return self.network.encoder_blocks[COLOUR_BLOCK].mre

    def get_trainable_roots(self):
        return [self.network.encoder_blocks[COLOUR_BLOCK]]

    def train(self, mode=True):
        super().train(False)
        for module in self.get_trainable_roots():
            module.train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        self.training = mode
        return self

    @contextmanager
    def _attach_colours(self, lab):
        self.colour_mre.lab = lab
        try:
            yield
        finally:
            self.colour_mre.lab = None

    def forward(self, xyz, rgb):
        batch, n_points, _ = xyz.shape
        offset = torch.arange(1, batch + 1, device=xyz.device, dtype=torch.int32) * n_points
        lab = convert_rgb_to_lab(rgb.float()).reshape(batch * n_points, 3)
        with self._attach_colours(lab):
            return self.network(xyz, None, offset)


def build_colour_weighted_pgd(variant, checkpoint=DEFAULT_CHECKPOINT, map_location="cpu"):
    return ColourWeightedPGD(load_pgd_network(checkpoint, map_location), variant=variant).eval()
