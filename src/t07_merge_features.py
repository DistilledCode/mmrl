import numpy as np
import chromadb
import pandas as pd
from itertools import islice
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

PROJ_DIR = os.environ.get("PROJ_DIR")

os.chdir(PROJ_DIR)

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def save_embeddings(collection, embeddings):
    for ind, batch in enumerate(batched(list(embeddings.keys()), 5000), start=1):
        collection.add(
            ids=[j for j in batch],
            embeddings=[embeddings[j] for j in batch],
        )
        print(f"\r[*] Added batch {ind:>05}", " " * 40, end="")
    print()
    print("[*] Number of IDs:", len(collection.get()["ids"]))
    print(
        "[*] Number of embeddings:",
        len(collection.get(include=["embeddings"])["embeddings"]),
    )


chroma_client = chromadb.PersistentClient(path="../embeddings")
features_collection_1 = chroma_client.get_or_create_collection(name="features_1")
features_collection_2 = chroma_client.get_or_create_collection(name="features_2")
desc_collection = chroma_client.get_collection(name="description")
print("[*] Loaded description vector collection")
title_collection = chroma_client.get_collection(name="title")
print("[*] Loaded title vector collection")
combined_collection = chroma_client.get_collection(name="combined")
print("[*] Loaded combined vector collection")
image_collection = chroma_client.get_collection(name="image")
print("[*] Loaded image vector collection")

print()

desc_ids = desc_collection.get()["ids"]
print("[*] Retrieved description ids")
title_ids = title_collection.get()["ids"]
print("[*] Retrieved title ids")
combined_ids = combined_collection.get()["ids"]
print("[*] Retrieved combined ids")
image_ids = image_collection.get()["ids"]
print("[*] Retrieved image ids")
imagedf = pd.read_parquet("../dump/final_images.parquet")
print()

print(f"[*] Posts with descriptions: {len(desc_ids)}")
print(f"[*] Posts with titles: {len(title_ids)}")
print(f"[*] Posts with combined: {len(combined_ids)}")
print(f"[*] Posts with images: {len(image_ids)}")
print()

s1 = set(features_collection_1.get()["ids"])
s2 = set(features_collection_2.get()["ids"])

assert s1 == s2

data = set(desc_ids).intersection(title_ids, combined_ids, image_ids)
data = data.intersection(set(imagedf["post_id"]))
print(f"[*] Total training samples: {len(data)}")
print(f"[*] Already stored samples: {len(s1)}")


data = data.difference(s1)
data = list(data)

print(f"[*] New samples to store: {len(data)}")
print()


desc_emdeddings = desc_collection.get(ids=data, include=["embeddings"])
desc_emdeddings = {
    i: j
    for i, j in zip(
        desc_emdeddings["ids"],
        desc_emdeddings["embeddings"],
    )
}
print(f"[*] Retrieved {len(desc_emdeddings)} description embeddings.")

title_emdeddings = title_collection.get(ids=data, include=["embeddings"])
title_emdeddings = {
    i: j
    for i, j in zip(
        title_emdeddings["ids"],
        title_emdeddings["embeddings"],
    )
}
print(f"[*] Retrieved {len(title_emdeddings)} title embeddings.")

combined_emdeddings = combined_collection.get(ids=data, include=["embeddings"])
combined_emdeddings = {
    i: j
    for i, j in zip(
        combined_emdeddings["ids"],
        combined_emdeddings["embeddings"],
    )
}
print(f"[*] Retrieved {len(combined_emdeddings)} combined embeddings.")

image_emdeddings = image_collection.get(ids=data, include=["embeddings"])
image_emdeddings = {
    i: j
    for i, j in zip(
        image_emdeddings["ids"],
        image_emdeddings["embeddings"],
    )
}
print(f"[*] Retrieved {len(image_emdeddings)} image embeddings.")
print()


print("[*] Creating feature vectors.")
df = pd.get_dummies(imagedf, columns=["subreddit"])
featuredf = df[
    [
        "post_id",
        "sub_members",
        "adult",
        "author_comment_score",
        "author_link_score",
        "subreddit_art",
        "subreddit_aww",
        "subreddit_earthporn",
        "subreddit_food",
        "subreddit_foodporn",
        "subreddit_itookapicture",
        "subreddit_natureisfuckinglit",
        "subreddit_oldschoolcool",
        "subreddit_painting",
        "subreddit_pics",
        "subreddit_roomporn",
        "subreddit_spaceporn",
    ]
]
featuredf = featuredf[featuredf["post_id"].isin(data)]
scaler = MinMaxScaler()
featuredf[["sub_members"]] = scaler.fit_transform(featuredf[["sub_members"]])
featuredf[["author_comment_score"]] = scaler.fit_transform(
    featuredf[["author_comment_score"]]
)
featuredf[["author_link_score"]] = scaler.fit_transform(
    featuredf[["author_link_score"]]
)
X = {}
Y = {}

print("[*] Concatenating posts embeddings.")
for ind, i in enumerate(data, start=1):
    _concatenated: list = desc_emdeddings[i]
    _concatenated.extend(title_emdeddings[i])
    _concatenated.extend(combined_emdeddings[i])
    _concatenated.extend(image_emdeddings[i])
    _concatenated = np.array(_concatenated)
    X[i] = _concatenated.tolist()
    Y[i] = np.concatenate(
        (
            _concatenated,
            featuredf[featuredf["post_id"] == i].values[0][1:].astype(float),
        )
    ).tolist()
    print(f"\r[*] [{ind:>06}] Concatenated embeddings for post {i}.", " " * 20, end="")
print()

print(f"[*] Saving embeddings with no post metadata.")
save_embeddings(features_collection_1, X)
print(f"[*] Saving embeddings with post metadata.")
save_embeddings(features_collection_2, Y)
