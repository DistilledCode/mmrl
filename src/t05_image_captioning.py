import os
import json
import pandas as pd
from pathlib import Path

PROJ_DIR = os.environ.get("PROJ_DIR")

os.chdir(PROJ_DIR)

os.environ["HF_HOME"] = "/mnt/f/irlab-gpu/.data/model/"
os.environ["HF_HUB_CACHE"] = "/mnt/f/irlab-gpu/.data/model/hub"
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

Image.MAX_IMAGE_PIXELS = None
print("[*] Loading model")
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("[*] Model loaded")

file_path = Path("../dump/image_captions.json")
if not file_path.exists():
    file_path.touch()
    print(f"File '{file_path}' created.")
    captions = dict()
else:
    print(f"File '{file_path}' already exists. Loading it.")
    with open(file_path, "r") as f:
        captions = json.load(f)
    print(f"Contains entry of {len(captions.keys())} posts")


def dump_captions(captions):
    with open("../dump/image_captions.json", "w") as f:
        json.dump(captions, f)
    print(f"[*] Dumped captions of {len(captions.keys())} posts!")


def gen_caption(img_path):
    image = Image.open(img_path).convert("RGB")
    prompt = "<grounding> Describe this image in detail:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=1000,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processed_text, _ = processor.post_process_generation(generated_text)
    return processed_text


imgdf = pd.read_parquet("../dump/final_images.parquet")

path = Path("../images")
count = 1
for file in path.iterdir():
    post_id = file.name.split(".")[0]
    if post_id in captions.keys():
        print(f"[~] {post_id} already exist, skipping it.")
        continue
    sub = imgdf[imgdf["post_id"] == post_id]["subreddit"].values[0]
    title = imgdf[imgdf["post_id"] == post_id]["title"].values[0]
    cap = gen_caption(file)
    cap = cap.split("Describe this image in detail:")[-1]
    captions[post_id] = {"Description": cap}
    print(
        f"[*] [{count:>06}]",
        f"[{sub.center(20)}]",
        f"[{post_id}]",
        cap[:65],
    )
    count += 1
    if count % 5 == 0:
        dump_captions(captions)
dump_captions(captions)
