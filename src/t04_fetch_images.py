from pathlib import Path
import requests
import concurrent.futures
import pandas as pd
from time import perf_counter
import os

PROJ_DIR = os.environ.get("PROJ_DIR")

os.chdir(PROJ_DIR)

st = perf_counter()
img_path = Path("../images/")
img_path.mkdir(exist_ok=True, parents=True)

imagedf = pd.read_parquet("../dump/filtered_posts.parquet")

img_ids = set(file.name.rsplit(".")[0] for file in img_path.iterdir())
print(f"Already have {len(img_ids)} images.")
print(f"Before trimming: {len(imagedf)} images.")
imagedf = imagedf[~imagedf["post_id"].isin(img_ids)]
print(f"After trimming : {len(imagedf)} images.")

count = 1
bad_urls = []


def fetch_image(postid: str, url: str, total: int):
    global count, bad_urls
    try:
        resp = requests.get(url)
        frmt = url.rsplit(".")[-1]
        if resp.status_code == 200:
            with open(f"../images/{postid}.{frmt}", "wb") as f:
                f.write(resp.content)
            if (postid, url) in bad_urls:
                bad_urls.remove((postid, url))
            print(
                f"[*] [{count:>06}/{total:>06}]",
                f"[{round(len(resp.content)/1e6,4):>05}]",
                f"[{len(bad_urls)}]",
                postid,
                url,
                resp.status_code,
            )
        else:
            if resp.status_code == 404:
                print(
                    f"[!] [{count:>06}/{total:>06}]",
                    "[???????????]",
                    postid,
                    "Status Code: 404, skipping it.",
                )
            else:
                bad_urls.append((postid, url))
                print(
                    f"[!] [{count:>06}/{total:>06}]",
                    "[???????????]",
                    f"[{len(bad_urls)}]",
                    postid,
                    url,
                    resp.status_code,
                )
    except Exception as e:
        bad_urls.append((postid, url))

        print(
            f"[!] [{count:>06}/{total:>06}]",
            "[???????????]",
            f"[{len(bad_urls)}]",
            postid,
            url,
            resp.status_code,
            e,
        )
    finally:
        count += 1


def save_images(df):
    if df.__class__ is list:
        df = pd.DataFrame(df, columns=["post_id", "url"])
    n = len(df)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(fetch_image, list(df["post_id"]), list(df["url"]), [n] * len(df))


save_images(imagedf)

print("Rechecking bad requests")
for i in range(5):
    print(f"[*] Retry number {i}")
    if len(bad_urls) == 0:
        break
    save_images(bad_urls)

print(f"[*] Bad URLs left: {len(bad_urls)}")


img_ids = set(file.name.rsplit(".")[0] for file in img_path.iterdir())
imagedf = pd.read_parquet("../dump/filtered_posts.parquet")

imagedf = imagedf[imagedf["post_id"].isin(img_ids)]
imagedf.sort_values(by="created", inplace=True)
imagedf = imagedf[
    [
        "created",
        "post_id",
        "title",
        "score",
        "url",
        "subreddit",
        "sub_members",
        "adult",
        "author_name",
        "author_comment_score",
        "author_link_score",
        "popularity",
    ]
]
imagedf.reset_index(drop=True, inplace=True)

print("[*] Dumping `final_images.parquet`")
imagedf.to_parquet("../dump/final_images.parquet")
tt = perf_counter() - st
print(f"[*] {__file__} took {tt//60} mins and {round(tt%60)} secs.")
