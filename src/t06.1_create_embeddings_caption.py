import os
import json
import chromadb

PROJ_DIR = os.environ.get("PROJ_DIR")

os.chdir(PROJ_DIR)

os.environ["HF_HOME"] = "/mnt/f/irlab-gpu/.data/model/"
os.environ["HF_HUB_CACHE"] = "/mnt/f/irlab-gpu/.data/model/hub"

from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer
from itertools import islice


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


chroma_client = chromadb.PersistentClient(path="../embeddings")
description = chroma_client.get_or_create_collection(name="description")

with open("../dump/image_captions.json", "r") as f:
    captions = json.load(f)

print(f"[*] Total ids: {len(captions)}")
print(f"[*] Total existing embeddings: {len(description.get()['ids'])}")
db_ids = set(description.get()["ids"])
new_captions = {i: captions[i] for i in captions if i not in db_ids}
print(f"[*] New ids to embed: {len(new_captions)}")

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
toklen = lambda x: len(tokenizer(x)["input_ids"])


def checkup(_captions, key):
    truncated = []
    count = 0
    for ind, i in enumerate(_captions, start=1):
        tklen = toklen(_captions[i][key])
        b = True if tklen > 8192 else False
        if b:
            truncated.append((i, tklen))
        count += bool(b)
        print(f"\r[{ind:>05}] [{i}] {tklen:>05} {b}", end="")
    print()
    print(f"[~] Number of sentences that will be truncated: {count}")
    return truncated


bad_captions = checkup(new_captions, "Description")
for bc in bad_captions:
    print(f"[~] Removing id {bc[0]}, toklen is {bc[1]}")
    new_captions.pop(bc[0])

print(f"[*] New good ids: {len(new_captions)}")


data = [new_captions[i]["Description"] for i in new_captions.keys()]
if len(data) > 0:
    model = BGEM3FlagModel("BAAI/bge-m3")
    embeddings = model.encode(data, batch_size=1, max_length=8192)["dense_vecs"]
else:
    print("[!] No new description to encode! Exiting.")
    exit(0)


for ind, batch in enumerate(
    batched(zip(new_captions.keys(), embeddings), 5000),
    start=1,
):
    description.add(
        ids=[j[0] for j in batch],
        embeddings=[j[1].tolist() for j in batch],
    )
    print(f"\r[*] Added batch {ind:>05}", end="")
print()

print("[*] Number of ids in vectorstore", len(description.get()["ids"]))

len(
    "[*] Number of embeddings in vectorstore",
    description.get(include=["embeddings"])["embeddings"],
)
