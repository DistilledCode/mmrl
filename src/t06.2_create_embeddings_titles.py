import os
import pandas as pd
import chromadb

PROJ_DIR = os.environ.get("PROJ_DIR")

os.chdir(PROJ_DIR)

os.environ["HF_HOME"] = "/mnt/f/irlab-gpu/.data/model/"
os.environ["HF_HUB_CACHE"] = "/mnt/f/irlab-gpu/.data/model/hub"

from itertools import islice


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer

chroma_client = chromadb.PersistentClient(path="../embeddings")
title_collection = chroma_client.get_or_create_collection(name="title")

imgdf = pd.read_parquet("../dump/final_images.parquet")

post_ids = imgdf["post_id"].tolist()

print(f"[*] Total ids: {len(post_ids)}")
print(f"[*] Total existing embeddings: {len(title_collection.get()['ids'])}")
db_ids = set(title_collection.get()["ids"])
new_ids = list(set(post_ids).difference(set(db_ids)))
print(f"[*] New ids to embed: {len(new_ids)}")

titles = imgdf[imgdf["post_id"].isin(new_ids)]["title"].tolist()
assert len(new_ids) == len(titles)
print(f"[*] New titles: {len(titles)}")
if len(titles) > 0:
    pass
else:
    print("[!] No new titles to encode! Exiting.")
    exit(0)

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
toklen = lambda x: len(tokenizer(x)["input_ids"])

model = BGEM3FlagModel("BAAI/bge-m3")
embeddings = model.encode(titles, batch_size=1, max_length=8192)["dense_vecs"]


for ind, batch in enumerate(batched(zip(post_ids, embeddings), 5000), start=1):
    title_collection.add(
        ids=[j[0] for j in batch],
        embeddings=[j[1].tolist() for j in batch],
    )
    print(f"\rAdded batch {ind:>05}", end="")
print()

print("[*] Number of ids in vectorstore", len(title_collection.get()["ids"]))

print(
    "[*] Number of embeddings in vectorstore",
    len(title_collection.get(include=["embeddings"])["embeddings"]),
)
