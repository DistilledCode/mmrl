import pandas as pd
from datetime import datetime
from time import perf_counter
import os

PROJ_DIR = os.environ.get("PROJ_DIR")

os.chdir(PROJ_DIR)

st = perf_counter()
print("[*] Loading post details")
df = pd.read_parquet("../dump/post_details.parquet")
print(f"[*] Number of post loaded: {len(df)}.")
print(f"[*] Filtering post with `subreddit==NaN` & non-image URLs")
df = df[~df["subreddit"].isna()]
df.reset_index(drop=True, inplace=True)


df = df[df["url"].str.startswith("https://i.redd.it/")]
print(f"[*] Number of posts left: {len(df)}.")
df["popularity"] = df["score"].div(df["sub_members"]) * 1000

print("[*] Discarding posts made in last 36 hours from training set.")
df = df[df["created"] < (datetime.now().timestamp() - 36 * 60 * 60)]
print("[*] Discarding posts with < 1 score.")
df = df[~(df["score"] < 1)]
print(f"[*] Number of posts left: {len(df)}.")


print("[*] Dumping to  `filtered_posts.parquet`")
df.to_parquet("../dump/filtered_posts.parquet")
tt = perf_counter() - st
print(f"[*] {__file__} took {tt//60} mins and {round(tt%60)} secs.")
