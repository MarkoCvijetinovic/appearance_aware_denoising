import sys
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from appearance_pgd import AppearancePGD
from torch_dataset import GSODataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PGD_ROOT = PROJECT_ROOT / "external" / "PGD"

sys.path.insert(0, str(PGD_ROOT))

from models.pgd import PGDModel

model = PGDModel.load_from_checkpoint(
    PGD_ROOT / "pretrained" / "PGD.ckpt",
    map_location="cpu",
    weights_only=False,
)

extended_model = AppearancePGD(model)

trainer = Trainer(
            accelerator='gpu',
            precision='16-mixed',
            devices='auto',
            num_nodes=1,
            deterministic=False,
            max_epochs=800,
            check_val_every_n_epoch=10,
            strategy=DDPStrategy()
        )

train_dataset = GSODataset("data/gso", "train", xyz_noise=0.02, rgb_noise=0.05)
val_dataset = GSODataset("data/gso", "val", xyz_noise=0.02, rgb_noise=0.05)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

#trainer.fit(extended_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

