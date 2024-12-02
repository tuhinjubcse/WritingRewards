# This script preprocesses the LAMP dataset (as shared in https://arxiv.org/abs/2409.14509)
# to compute z-scores for creativity scores and save the results to a new JSON file.
# and to add an editor-based dataset split
from collections import Counter
import numpy as np, json

data_fn = "data/LAMP-train-val-test.json"

with open(data_fn, "r") as f:
    data = json.load(f)

for d in data:
    d['editor'] = d['id'].split('_')[0]
    d['creativity_pre_score'] = int(d['creativity_scores'][0])
    d['creativity_post_score'] = int(d['creativity_scores'][1])

editors = list(set([d["editor"] for d in data]))

editor_scores = {editor: [] for editor in editors}
editor_pre_scores = {editor: [] for editor in editors}
editor_post_scores = {editor: [] for editor in editors}

for d in data:
    editor_scores[d["editor"]].append(d["creativity_pre_score"])
    editor_scores[d["editor"]].append(d["creativity_post_score"])
    editor_pre_scores[d["editor"]].append(d["creativity_pre_score"])
    editor_post_scores[d["editor"]].append(d["creativity_post_score"])

# print avg and std of creativity scores per editor
editor_means, editor_stds = {}, {}
for editor in editors:
    editor_means[editor] = np.mean(editor_scores[editor])
    editor_stds[editor] = np.std(editor_scores[editor])

print(f"Editors: {editors}")

for editor in sorted(editors, key=lambda x: editor_means[x]):
    print(f"{editor}: {editor_means[editor]:.2f} ± {editor_stds[editor]:.2f}")

# Calculate a creativity z-score for each editor
for d in data:
    d["creativity_z_score_pre"] = (d["creativity_pre_score"] - editor_means[d["editor"]]) / editor_stds[d["editor"]]
    d["creativity_z_score_post"] = (d["creativity_post_score"] - editor_means[d["editor"]]) / editor_stds[d["editor"]]

min_z_score = min([d["creativity_z_score_pre"] for d in data] + [d["creativity_z_score_post"] for d in data])
max_z_score = max([d["creativity_z_score_pre"] for d in data] + [d["creativity_z_score_post"] for d in data])

# renormalize z-scores to be between 0 and 10
for d in data:
    d["creativity_z_score_pre"] = 10 * (d["creativity_z_score_pre"] - min_z_score) / (max_z_score - min_z_score)
    d["creativity_z_score_post"] = 10 * (d["creativity_z_score_post"] - min_z_score) / (max_z_score - min_z_score)
    d["creativity_z_score_pre_int"] = round(d["creativity_z_score_pre"])
    d["creativity_z_score_post_int"] = round(d["creativity_z_score_post"])
    d["creativity_z_score_diff"] = d["creativity_z_score_post_int"] - d["creativity_z_score_pre_int"]

print("--------------------------------")
print("Post normalization")

# New mean and std of creativity z-scores
for editor in editors:
    editor_samples = [d for d in data if d["editor"] == editor]
    editor_scores = [d['creativity_z_score_pre_int'] for d in editor_samples] + [d['creativity_z_score_post_int'] for d in editor_samples]
    print(f"{editor}: {np.mean(editor_scores):.2f} ± {np.std(editor_scores):.2f}")

# calculate num samples where either pre or post scores have changed
num_changed = len([d for d in data if d['creativity_z_score_post_int'] != d['creativity_post_score'] or d['creativity_z_score_pre_int'] != d['creativity_pre_score']])
print(f"Number of samples where either pre or post scores have changed: {num_changed} / {len(data)}")

# Now deal with editor-based splits
# if editor number is even, put in test set, otherwise put in train set
writer_split_key = "editor_split"
for d in data:
    editor_num = int(d["editor"][1:])
    d[writer_split_key] = "test" if editor_num % 2 == 0 else "train"

print(f"Editor split: {Counter([d[writer_split_key] for d in data])}")

with open(data_fn, "w") as f:
    json.dump(data, f, indent=4)
