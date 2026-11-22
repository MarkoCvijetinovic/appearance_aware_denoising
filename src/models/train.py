import sys
from pathlib import Path
from appearance_pgd import AppearancePGD
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

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
# trainer.fit(extended_model, ckpt_path=PGD_ROOT / "pretrained" / "PGD.ckpt")

