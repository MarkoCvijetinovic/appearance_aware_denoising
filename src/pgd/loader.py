import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PGD_ROOT = PROJECT_ROOT / "external" / "PGD"
DEFAULT_CHECKPOINT = PGD_ROOT / "pretrained" / "PGD.ckpt"


def add_pgd_to_path():
    if str(PGD_ROOT) not in sys.path:
        sys.path.append(str(PGD_ROOT))


def load_pgd_network(checkpoint=DEFAULT_CHECKPOINT, map_location="cpu"):
    add_pgd_to_path()
    from models.feature import FeatureExtraction

    state = torch.load(checkpoint, map_location=map_location, weights_only=False)["state_dict"]
    prefix = "feature_nets."
    network = FeatureExtraction()
    network.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    return network


class FrozenPGD(torch.nn.Module):
    """Unmodified PGD with the same call form as our method: model(xyz, rgb) -> (displacement, commitment).

    Colour is ignored; the adapter only computes the offset that PGD's network needs.
    """
    # Schnittstelle
    last_gate = None

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, xyz, rgb=None):
        # xyz hat die Form (Anzahl Patches, 1000 Punkte, 3)
        # rgb sind die Farben dazu, in derselben Form
        batch, n_points, _ = xyz.shape
        offset = torch.arange(1, batch + 1, device=xyz.device, dtype=torch.int32) * n_points
        # offset sagt, wo jeder Patch in der Liste endet
        return self.network(xyz, None, offset)


def get_frozen_pgd(checkpoint=DEFAULT_CHECKPOINT, map_location="cpu"):
    """Load the released PGD as FrozenPGD: no trainable weights, in eval mode.

    Reference for our method, which must give exactly the same result before training.
    """
    # lädt das Netz, steckt es in den Adapter und stellt zwei Dinge ein
        # requires_grad_(False): Keine Gewichte sind trainierbar, PGD bleibt eingefroren
        # eval(): setzt bei jedem Teil des Netzes das Attribut training auf False
    return FrozenPGD(load_pgd_network(checkpoint, map_location)).requires_grad_(False).eval()