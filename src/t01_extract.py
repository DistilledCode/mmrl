import os
import pathlib
import pandas as pd
from time import perf_counter
import json
from pathlib import Path

PROJ_DIR = os.environ.get("PROJ_DIR")

os.chdir(PROJ_DIR)

for dir in ("../data", "../dump", "../images", "../models", "../embeddings"):
    Path(dir).mkdir(parents=True, exist_ok=True)


FILE_PATH = Path("../dump/processed_dataframes.json")
DATA_DIRECTORY = "../data/"
DF_PATH = Path("../dump/post_ids.parquet")
TARGET_SUBS = [
    "art",
    "aww",
    "food",
    "pics",
    "painting",
    "foodporn",
    "roomporn",
    "earthporn",
    "spaceporn",
    "oldschoolcool",
    "itookapicture",
    "natureisfuckinglit",
]


def get_processed_dfs():
    if not FILE_PATH.exists():
        FILE_PATH.touch()
        print(f"File '{FILE_PATH}' created.")
        proc_dfs = []
    else:
        print(f"[*] File '{FILE_PATH}' already exists. Loading it.")
        with open(FILE_PATH, "r") as f:
            proc_dfs = json.load(f)
        print(f"[*] Already have processed {len(proc_dfs)} dfs")
    return proc_dfs


def get_new_dfs(proc_dfs):
    dfs = []
    proc_dfs_set = set(proc_dfs)
    for ind, path in enumerate(pathlib.Path(DATA_DIRECTORY).glob("*.parquet"), start=1):
        if path.name in proc_dfs_set:
            print(f"\r[*] [{ind:>06}] Already processed {path}", " " * 20, end="")
            continue
        df = pd.read_parquet(path)
        subreddit_map = (
            df["permalink"]
            .str.lower()
            .str.split("/", expand=True)[2]
            .str.lower()
            .isin(TARGET_SUBS)
        )
        df = df[subreddit_map]
        dfs.append(df)
        print(f"\r[*] [{ind:>06}] [{len(df):>04}] {path}", " " * 20, end="")
        proc_dfs.append(path.name)
    print()
    print(f"[*] Processed {len(dfs)} new dataframes.")
    return dfs, proc_dfs


def process_dfs(dfs):
    if not dfs:
        print("[?] No new dfs to process. Exiting.")
        return None
    df = pd.concat(dfs, axis=0)

    print("[*] Expanding `permalink` text into columns.")
    df = df["permalink"].str.lower().str.split("/", expand=True)[[2, 4]]

    df.columns = ("subreddit", "post_id")
    df.drop_duplicates("post_id", inplace=True)
    print(f"[*] Dropped duplicates, {len(df)} rows left.")
    df.reset_index(drop=True, inplace=True)

    if DF_PATH.exists():
        df = pd.concat((df, pd.read_parquet("../dump/post_ids.parquet")), axis=0)
    df.drop_duplicates("post_id", inplace=True)
    return df


def dump_data(df, proc_dfs):
    print("[*] Dumping to `../dump/post_ids.parquet`")
    df.to_parquet("../dump/post_ids.parquet")
    print(f"[*] Total possible target posts: {len(df)}")
    with open(FILE_PATH, "w") as f:
        print("Saving processed dataframes list.")
        json.dump(proc_dfs, f)
        print(f"[*] Total dataframes processed so far: {len(proc_dfs)}")
    print("[*] Post distribution by subreddit:")
    print(
        df.groupby("subreddit")["post_id"]
        .aggregate("count")
        .sort_values(ascending=False)
    )


def time_elapsed(st):
    tt = perf_counter() - st
    print(f"[*] {__file__} took {tt//60} mins and {round(tt%60)} secs.")


def main():
    st = perf_counter()
    proc_dfs = get_processed_dfs()
    dfs, proc_dfs = get_new_dfs(proc_dfs)
    df = process_dfs(dfs)
    if df is not None:
        dump_data(df, proc_dfs)
    time_elapsed(st)


if __name__ == "__main__":
    main()
