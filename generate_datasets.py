import argparse, json, random

parser = argparse.ArgumentParser()
parser.add_argument("--data_fn", type=str, default="data/LAMP-extended.json")
parser.add_argument("--min_score_threshold", type=int, default=1.0)
parser.add_argument("--pairwise_prompt_fn", type=str, default="prompts/pairwise_pref.txt")
parser.add_argument("--include_pairwise_pref", action="store_true")
parser.add_argument("--include_reward_scoring", action="store_true")
parser.add_argument("--reward_prompt_fn", type=str, default="prompts/reward_calc.txt")

args = parser.parse_args()

assert args.include_pairwise_pref or args.include_reward_scoring, "Must include either pairwise preference or reward scoring"

added_param = "_reward" if args.include_reward_scoring else ""

out_files = f"data/finetune_{'P' if args.include_pairwise_pref else ''}{'R' if args.include_reward_scoring else ''}_[SPLIT].json"

with open(args.data_fn, "r") as f:
    lamp_data = json.load(f)

train_samples, test_samples = [], []

with open(args.pairwise_prompt_fn, "r") as f:
    pairwise_prompt = f.read()

with open(args.reward_prompt_fn, "r") as f:
    reward_prompt = f.read()

train_pairwise, test_pairwise = [], []
if args.include_pairwise_pref:
    for d in lamp_data:
        diff = abs(d['creativity_z_score_post'] - d['creativity_z_score_pre'])
        if diff < args.min_score_threshold:
            continue

        # For now, include both orderings everytime
        sample1 = {"original_id": d["id"], "split": d["reward-split"], "source": d["source"], "type": d["type"], "sample_type": "pairwise", "paragraph1": d["preedit"], "paragraph2": d["postedit"], "reference_preference": "2"}
        sample1["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", d["preedit"]).replace("[[PARAGRAPH2]]", d["postedit"])
        sample1["output"] = '{"preference": "2"}'

        sample2 = {"original_id": d["id"], "split": d["reward-split"], "source": d["source"], "type": d["type"], "sample_type": "pairwise", "paragraph1": d["postedit"], "paragraph2": d["preedit"], "reference_preference": "1"}
        sample2["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", d["postedit"]).replace("[[PARAGRAPH2]]", d["preedit"])
        sample2["output"] = '{"preference": "1"}'
        
        if d["reward-split"] == "train":
            train_pairwise.append(sample1)
            train_pairwise.append(sample2)
        elif d["reward-split"] == "test":
            test_pairwise.append(sample1)
            test_pairwise.append(sample2)

train_reward, test_reward = [], []
if args.include_reward_scoring:
    for d in lamp_data:
        sample3 = {"original_id": d["id"], "split": d["reward-split"], "source": d["source"], "type": d["type"], "sample_type": "reward", "paragraph": d["preedit"], "zscore": d["creativity_z_score_pre_int"]}
        sample3["text_input"] = reward_prompt.replace("[[PARAGRAPH]]", d["preedit"])
        sample3["output"] = '{"score": '+str(d["creativity_z_score_pre_int"])+'}'

        sample4 = {"original_id": d["id"], "split": d["reward-split"], "source": d["source"], "type": d["type"], "sample_type": "reward", "paragraph": d["postedit"], "zscore": d["creativity_z_score_post_int"]}
        sample4["text_input"] = reward_prompt.replace("[[PARAGRAPH]]", d["postedit"])
        sample4["output"] = '{"score": '+str(d["creativity_z_score_post_int"])+'}'

        if d["reward-split"] == "train":
            train_reward.append(sample3)
            train_reward.append(sample4)
        elif d["reward-split"] == "test":
            test_reward.append(sample3)
            test_reward.append(sample4)

for i, d in enumerate(train_pairwise):
    d["id"] = f"train-pairwise-{i}"

for i, d in enumerate(test_pairwise):
    d["id"] = f"test-pairwise-{i}"

for i, d in enumerate(train_reward):
    d["id"] = f"train-reward-{i}"

for i, d in enumerate(test_reward):
    d["id"] = f"test-reward-{i}"

train_samples = train_pairwise + train_reward
test_samples = test_pairwise + test_reward

print(f"Train samples: {len(train_samples)}")
print(f"Test samples: {len(test_samples)}")

random.shuffle(train_samples)
random.shuffle(test_samples)

with open(out_files.replace("[SPLIT]", "train"), "w") as f:
    json.dump(train_samples, f, indent=2)

with open(out_files.replace("[SPLIT]", "test"), "w") as f:
    json.dump(test_samples, f, indent=2)
