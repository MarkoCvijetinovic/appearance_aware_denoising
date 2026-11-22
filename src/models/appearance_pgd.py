import pytorch_lightning as pl
import torch
from pytorch3d.loss import chamfer_distance

class AppearancePGD(pl.LightningModule):
    def __init__(self, pgd):
        super().__init__()

        self.pgd = pgd
        self.pgd.requires_grad_(False)
        self.pgd.eval()

        # Adding colour module to PGD
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(3 + 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU()
        )

    def forward(self, noisy_xyz, noisy_rgb):
        # Run frozen PGD.
        with torch.no_grad():
            base_xyz = self.pgd.patch_based_denoise(
                pcl_noisy=noisy_xyz,
                patch_size=1000,
                seed_k=6,
                seed_k_alpha=10,
            )

        # Predict a residual correction.
        correction = self.adapter(torch.cat([base_xyz, noisy_rgb], dim=-1))

        return base_xyz + correction

    def training_step(self, batch, batch_idx):
        predicted_xyz = self(batch["noisy_xyz"], batch["noisy_rgb"])

        loss, _ = chamfer_distance(predicted_xyz, batch["clean_xyz"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predicted_xyz = self(batch["noisy_xyz"], batch["noisy_rgb"])

        loss, _ = chamfer_distance(predicted_xyz, batch["clean_xyz"])
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.adapter.parameters(), lr=1e-3)