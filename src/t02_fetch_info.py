import praw
import pandas as pd
import concurrent.futures
import bs4
import requests
import re
import numpy as np
from time import perf_counter
from datetime import datetime
import os

PROJ_DIR = os.environ.get("PROJ_DIR")

os.chdir(PROJ_DIR)


st = perf_counter()
print("[*] Loading Reddit Credentials")
reddit = praw.Reddit("bot1", user_agent="just-passing-by")
print(f"[*] {reddit.read_only}")
print(f"[*] {reddit.user.me()}")

df = pd.read_parquet("../dump/post_ids.parquet")
print(f"[*] `post_ids.parquet` loaded with {len(df)} rows")

ddf = pd.read_parquet("../dump/post_details.parquet")
print(f"[*] `post_details.parquet` loaded with {len(ddf)} rows")

target = df[~df["post_id"].isin(ddf["post_id"])]
print(f"[*] {len(target)} new ids to fetch.")

if len(target) == 0:
    print("[!] No new IDs to fetch. Exiting.")
    exit(0)


def parse_gallery(link):
    resp = requests.get(link)
    soup = bs4.BeautifulSoup(resp.text, "html.parser")
    anchors = soup.find_all("a", href=True)
    reason = "Bad Request" if resp.status_code != 200 else "Exhausted all <a>"
    for i in anchors:
        match = re.search(r"^(https://.*\.jpg)\?", i["href"])
        if match:
            return match.group(0)[:-1].replace("preview", "i")
    return f"[!] Error while parsing gallery: {reason}"


count = 1
post_details = []


def fetch_url(post_id, n):
    global count, post_details
    t = datetime.now().strftime("%b %d %a %X")
    try:
        post = reddit.submission(post_id)
        url = post.url
        created = post.created_utc
        title = post.title
        adult = post.over_18
        score = post.score
        sub_members = post.subreddit.subscribers
        author = post.author
        if author is None:
            author_name = "Deleted User"
        else:
            author_name = post.author.name
        try:
            author_comment_score = post.author.comment_karma
            author_link_score = post.author.link_karma
        except Exception:
            author_comment_score = 0
            author_link_score = 0
        if "www.reddit.com/gallery" in url:
            url = parse_gallery(url)
            print(f"[*][{count:>06}/{n:>06}] {t}", post_id, "[GALLERY]".rjust(14), url)
        else:
            print(f"[*] [{count:>06}/{n:>06}] {t}", post_id, "[NON-GALLERY]".rjust(14))
    except Exception as e:
        url = np.NaN
        created = np.NaN
        title = np.NaN
        adult = np.NaN
        score = np.NaN
        sub_members = np.NaN
        author_name = np.NaN
        author_comment_score = np.NaN
        author_link_score = np.NaN
        print(f"[!] [{count:>06}/{n:>06}] {t}", post_id, "[???????????]".rjust(14), e)
    count += 1
    post_details.append(
        {
            "post_id": post_id,
            "title": title,
            "url": url,
            "created": created,
            "adult": adult,
            "score": score,
            "sub_members": sub_members,
            "author_name": author_name,
            "author_comment_score": author_comment_score,
            "author_link_score": author_link_score,
        }
    )


def get_details(df):
    global post_details
    post_details = []
    n = len(df)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(fetch_url, list(df["post_id"]), [n] * len(df))
    return pd.DataFrame(post_details)


ddf = get_details(target)

ddf = pd.merge(target, ddf, how="inner", on="post_id")

print("[*] Retrying on unsuccessful fetches.")
for i in range(3):
    print(f"[*] Retry number {i+1}")
    retries = ddf[ddf["url"].isna()]
    print(f"[*] Retrying for {len(retries)} ids.")
    if len(retries) == 0:
        print(f"[*] All post fetched! Breaking the loop.")
        break
    ddf = ddf[~ddf["url"].isna()]
    retried = get_details(retries)
    ddf = pd.concat((ddf, retried), axis=0)


ddf.reset_index(drop=True, inplace=True)
target.reset_index(drop=True, inplace=True)

print(f"[*] Final new ids fetch count: {len(ddf)}")

bad_ids = target[ddf["url"].isna()]

df = df[
    ~df["post_id"].isin(
        pd.merge(
            df,
            bad_ids,
            how="inner",
            on="post_id",
        )["post_id"]
    )
]

ddf["url"].isna().value_counts(), ddf["subreddit"].isna().value_counts()

ddf = ddf[~ddf["url"].isna()]
ddf = ddf[~ddf["subreddit"].isna()]

ddf = pd.concat((ddf, pd.read_parquet("../dump/post_details.parquet")), axis=0)
print(f"[*] Final total fetched ids count: {len(ddf)}")


assert ddf["post_id"].nunique() == len(ddf)

print("[*] Dumping dataframes")
ddf.to_parquet("../dump/post_details.parquet")
df.to_parquet("../dump/post_ids.parquet")

tt = perf_counter() - st
print(f"[*] {__file__} took {tt//60} mins and {round(tt%60)} secs.")
