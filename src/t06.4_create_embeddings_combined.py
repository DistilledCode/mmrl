import os
import json
from pathlib import Path
from PIL import Image

PROJ_DIR = os.environ.get("PROJ_DIR")

os.chdir(PROJ_DIR)

Image.MAX_IMAGE_PIXELS = None

import chromadb

os.environ["HF_HOME"] = "/mnt/f/irlab-gpu/.data/model/"
os.environ["HF_HUB_CACHE"] = "/mnt/f/irlab-gpu/.data/model/hub"
from keras_nlp import models

tokenizer = models.MistralTokenizer.from_preset("mistral_7b_en")
import torch
from FlagEmbedding.visual.modeling import Visualized_BGE

print("[*] Modules imported")

path = Path("../images")
image_pids = [i.name.split(".")[0] for i in list(path.iterdir())]
print(f"Total post ID from images: {len(image_pids)}")

with open("../dump/image_captions.json", "r") as f:
    captions = json.load(f)
print(f"Total post ID from captions: {len(captions.keys())}")

target_ids = set(image_pids).intersection(set(captions.keys()))
print(f"Length of intersection of both: {len(target_ids)}")


chroma_client = chromadb.PersistentClient(path="../embeddings")
combined = chroma_client.get_or_create_collection(name="combined")
# chroma_client.delete_collection(name="combined")
print(f"Total ids: {len(target_ids)}")
print(f"Total embeddings: {len(combined.get()['ids'])}")
db_ids = set(combined.get()["ids"])
new_target_ids = target_ids.difference(set(db_ids))
print(f"New ids: {len(new_target_ids)}")
if len(new_target_ids) == 0:
    print("Now new data points to embed. DB is utpo date.")


model = Visualized_BGE(
    model_name_bge="BAAI/bge-m3",
    model_weight="/mnt/f/irlab-gpu/.data/model/hub/Visualized_m3.pth",
)
print("[*] Model loaded")


from itertools import islice


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def save_embeddings(embeddings):
    for ind, batch in enumerate(batched(list(embeddings.keys()), 5000), start=1):
        combined.add(
            ids=[j for j in batch],
            embeddings=[embeddings[j] for j in batch],
        )
        print(f"\r[*] Added batch {ind:>05}", " " * 40, end="")
    print()
    print("[*] Number of ID Length :", len(combined.get()["ids"]))
    print(
        "[*] Number of embeddings:",
        len(combined.get(include=["embeddings"])["embeddings"]),
    )


embeddings = {}
done = 0
with torch.no_grad():
    for ind, file in enumerate(sorted(list(path.iterdir())), start=1):
        _post_id = file.name.split(".")[0]
        if _post_id not in new_target_ids:
            print(
                f"\r[~] [{ind:>05}] [{file.name}] Post ID not common or enbedding already exist, skipping it.",
                end="",
            )
            continue
        descrip = captions[_post_id]["Description"]
        desc_toklen = len(tokenizer(descrip))
        if desc_toklen > 8192:
            print(f"")
            print(
                f"\r[!] [{ind:>05}] [{file.name}] toklen {desc_toklen} is more than limit.",
                " " * 30,
                end="",
            )
            continue
        try:
            embedding = model.encode(image=file, text=descrip)
        except Exception as e:
            print(f"\n[!] [{ind:>05}] Exception for {file.name}", " " * 50, end="")
            print(f"\n[!] [{ind:>05}] error: {str(e)}", " " * 50, end="")
            continue
        embeddings[_post_id] = embedding.tolist()[0]
        print(f"\r[*] [{ind:>05}] Embedding created for {file.name}", " " * 50, end="")
        done += 1
        if done % 100 == 0:
            print()
            save_embeddings(embeddings)
            print(f"\r[*] Saved embeddings", " " * 50, end="")
            embeddings = {}
save_embeddings(embeddings)
