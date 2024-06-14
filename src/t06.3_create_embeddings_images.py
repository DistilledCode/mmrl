import os
import torch
import chromadb
from PIL import Image
from pathlib import Path

PROJ_DIR = os.environ.get("PROJ_DIR")

os.chdir(PROJ_DIR)

os.environ["HF_HOME"] = "/mnt/f/irlab-gpu/.data/model/"
os.environ["HF_HUB_CACHE"] = "/mnt/f/irlab-gpu/.data/model/hub"
Image.MAX_IMAGE_PIXELS = None

import timm

path = Path("../images")
chroma_client = chromadb.PersistentClient(path="../embeddings")
image_collection = chroma_client.get_or_create_collection(name="image")

post_ids = [i.name.split(".")[0] for i in list(path.iterdir())]

print(f"[*] Total ids: {len(post_ids)}")
print(f"[*] Total existing embeddings: {len(image_collection.get()['ids'])}")
db_ids = set(image_collection.get()["ids"])
new_ids = list(set(post_ids).difference(set(db_ids)))
print(f"[*] New ids to embed: {len(new_ids)}")

from itertools import islice


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = timm.create_model(
    "vit_large_patch16_384.augreg_in21k_ft_in1k",
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
).to(device)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
embeddings = {}
for ind, file in enumerate(sorted(list(path.iterdir())), start=1):
    _post_id = file.name.split(".")[0]
    if _post_id not in new_ids:
        print(
            f"\r[~] [{ind:>05}] Embedding for {file.name} exist, skipping it.",
            " " * 20,
            end="",
        )
        continue
    img = Image.open(file).convert("RGB")
    output = model.forward_features(transforms(img).unsqueeze(0).to(device))
    embedding = model.forward_head(output, pre_logits=True)
    embeddings[_post_id] = embedding.tolist()[0]
    print(f"\r[*] [{ind:>05}] Embedding created for {file.name}", " " * 20, end="")
print()

for ind, batch in enumerate(batched(list(embeddings.keys()), 5000), start=1):
    image_collection.add(
        ids=[j for j in batch],
        embeddings=[embeddings[j] for j in batch],
    )
    print(f"\rAdded batch {ind:>05}", end="")
print()

print("[*] Number of ids in vectorstore", len(image_collection.get()["ids"]))

print(
    "[*] Number of embeddings in vectorstore",
    len(image_collection.get(include=["embeddings"])["embeddings"]),
)
