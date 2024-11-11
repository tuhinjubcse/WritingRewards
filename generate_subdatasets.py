import argparse, json, random
import random

parser = argparse.ArgumentParser()
parser.add_argument("--data_fn", type=str, default="data/LAMP-train-val-test.json")
parser.add_argument("--split", type=str, default="test")

args = parser.parse_args()

with open(args.data_fn) as f:
    data = json.load(f)

data = [d for d in data if d["data-split"] == args.split]

with open("prompts/pairwise_pref.txt", "r") as f:
    pairwise_prompt = f.read()

def keep_edits(d, N_keep=2):
    edits = [e for e in d["fine_grained_edits"] if e["originalText"] in d["preedit"]]

    if N_keep == "all":
        N_keep = len(edits)

    if len(edits) < N_keep:
        return None

    random.shuffle(edits)
    keep_edits = edits[:N_keep]
    output_text = d["preedit"]
    for edit in keep_edits:
        output_text = output_text.replace(edit["originalText"], edit["editedText"])
    return output_text

N_keeps = [1, 2, 3, 4, 5, 6, 7, "all"]
datasets = {N_keep: [] for N_keep in N_keeps}

for d_idx, d in enumerate(data):
    for N_keep in N_keeps:
        subedit = keep_edits(d, N_keep)
        if subedit is not None:
            preedit = d["preedit"]
            sample = {"original_id": d["id"], "split": d["data-split"], "source": d["source"], "type": d["type"], "sample_type": f"pairwise-P{N_keep}"}
            if d_idx % 2 == 0:
                sample.update({"paragraph1": preedit, "paragraph2": subedit, "reference_preference": "2", "output": '{"preference": "2"}'})
            else:
                sample.update({"paragraph1": subedit, "paragraph2": preedit, "reference_preference": "1", "output": '{"preference": "1"}'})
            sample["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", sample["paragraph1"]).replace("[[PARAGRAPH2]]", sample["paragraph2"])
            datasets[N_keep].append(sample)


for N_keep in N_keeps:
    for i, d in enumerate(datasets[N_keep]):
        d["id"] = f"pairwise-P{N_keep}-{i}"

    with open(f"data/subedits_P{N_keep}_{args.split}.json", "w") as f:
        json.dump(datasets[N_keep], f)
