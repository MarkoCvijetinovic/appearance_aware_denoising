
from datasets import load_dataset
from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
out = PROJECT_ROOT / "data" / "catalogue"
thumb_dir = out / "thumbnails"
thumb_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    "suvadityamuk/google-scanned-objects",
    split="train",
    streaming=True,
)

records = []

for sample in ds:
    meta = sample["json"]

    object_id = str(meta["object_id"])
    name = str(meta.get("name") or object_id)
    category = str(meta.get("category") or "Unknown")


    image = sample["thumbnail_0.jpg"]
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    image.save(thumb_dir / f"{object_id}.jpg")

    records.append({
        "object_id": object_id,
        "name": name,
        "category": category,
    })

with open(out / "objects.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

# Avoid one enormous, difficult-to-read image.
per_page = 25
columns = 5

for start in range(0, len(records), per_page):
    page = records[start:start + per_page]
    rows = math.ceil(len(page) / columns)

    fig, axes = plt.subplots(
        rows, columns, figsize=(20, rows * 2.3),
        squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, record in zip(axes.flat, page):
        img = Image.open(thumb_dir / f"{record['object_id']}.jpg")
        ax.imshow(img)
        ax.set_title(
            f"{record['name'][:21]}\n"
            f"{record['category'][:24]}\n"
            f"ID: {record['object_id']}",
            fontsize=7,
        )

    fig.tight_layout()
    page_index = start // per_page + 1
    fig.savefig(out / f"catalogue_{page_index:02d}.jpg", dpi=150)
    plt.close(fig)

print(f"Catalogued {len(records)} objects in {out}")
