import datetime
from pathlib import Path
import praw
import pandas as pd
from prawcore.exceptions import RequestException, PrawcoreException
from time import sleep
import os

os.environ["SCRAPPED_DATA_DIR"] = os.getcwd() + "/data/"

import signal


def signal_handler(sig, frame):
    print("Python scrip received SIGINT signal. Terminating Python script...")
    dump_data(comments)
    exit(0)


# Set up the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)


reddit = praw.Reddit("bot2", user_agent="just-passing-by")


comments_per_df = 100_000
cols = ["time_created", "body", "author", "permalink"]

print(f"{reddit.user.me()=}")
print(f"{reddit.read_only=}")


path = Path("./data/")
path.mkdir(parents=True, exist_ok=True)


def dump_data(comments: list):
    if len(comments) == 0:
        print(f"Comment list empty. No comments to dump.")
        return
    dfname = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dfpath = f"./data/{dfname}.parquet"
    pd.DataFrame(comments, columns=cols).to_parquet(dfpath)
    print(f"[*] Saved df at {dfpath} with {len(comments)} records.")
    comments.clear()


count = 0
comments = []
first_run = True
start_time = 0
err_count = 0
with open("subreddits_small.txt", "r") as f:
    subs = set(i.strip().lower()[3:] for i in f.readlines())

print(f"{len(subs)=}")


def run_scrapper():
    global count
    global comments
    global first_run
    global start_time
    global err_count
    for comment in reddit.subreddit("all").stream.comments():
        if (
            comment.permalink.lower()[1:].split("/")[1] in subs
            and comment.author is not None
        ):
            #        ) and comment.body.isascii():
            #            count +=1
            # if :
            count += 1
            comments.append(
                {
                    "time_created": comment.created_utc,
                    "body": comment.body,
                    "author": comment.author.name,
                    "permalink": comment.permalink,
                },
            )
            latest = comments[-1]
            if first_run is True:
                start_time = comments[0]["time_created"]
                first_run = False
            spd_ = (count / (latest["time_created"] - start_time + 1e-10)) * 60
            ts = str(datetime.datetime.fromtimestamp(latest["time_created"]))
            print(
                f"[{count:>04}] [{ts}] [{round(spd_,3):>7}] [{err_count:>02}] "
                f'{comment.id}: {latest["permalink"]}'
            )
            if count % comments_per_df == 0:
                dump_data(comments)


try:
    run_scrapper()
except KeyboardInterrupt:
    err_count += 1
    answer = input("save the data? [y/n] ").lower()
    if answer == "y":
        dump_data(comments)
    else:
        print("Exiting!")
except RequestException:
    err_count += 1
    print(
        "[!] Internet Connection Error."
        "Retrying after 2 minutes. Dumping the existing files."
    )
    dump_data(comments)
    sleep(120)
    run_scrapper()

except PrawcoreException as e:
    err_count += 1
    print(
        "[!] Prawcore error encountered."
        "Retrying after 20 secs. Dumping the existing files."
    )
    print("[!] ", f"{str(e)}".center(80))
    dump_data(comments)
    sleep(20)
    run_scrapper()
except Exception as e:
    err_count += 1
    print(
        "[!] Generic Python Error."
        "Retrying after 10 secs. Dumping the existing files."
    )
    print("[!] ", f"{str(e)}".center(80))
    dump_data(comments)
    sleep(10)
    run_scrapper()
finally:
    dump_data(comments)
