import argparse, json, random

parser = argparse.ArgumentParser()
parser.add_argument("--data_fn", type=str, default="data/LAMP-extended.json")
parser.add_argument("--min_score_threshold", type=int, default=1.0)
parser.add_argument("--pairwise_prompt_fn", type=str, default="prompts/pairwise_pref.txt")
parser.add_argument("--include_reward_scoring", action="store_true")
parser.add_argument("--reward_prompt_fn", type=str, default="prompts/reward_calc.txt")


args = parser.parse_args()

added_param = "_reward" if args.include_reward_scoring else ""

out_files = f"data/finetune_pairwise{'_reward' if args.include_reward_scoring else ''}_[SPLIT].json"

with open(args.data_fn, "r") as f:
    data = json.load(f)

train_samples, test_samples = [], []

with open(args.pairwise_prompt_fn, "r") as f:
    pairwise_prompt = f.read()

with open(args.reward_prompt_fn, "r") as f:
    reward_prompt = f.read()

for d in data:
    diff = abs(d['creativity_z_score_post'] - d['creativity_z_score_pre'])
    if diff < args.min_score_threshold:
        continue

    # For now, include both orderings everytime
    sample1 = {"original_id": d["id"], "split": d["reward-split"], "source": d["source"], "type": d["type"]}
    sample1["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", d["preedit"]).replace("[[PARAGRAPH2]]", d["postedit"])
    sample1["output"] = '{"preference": "2"}'

    sample2 = {"original_id": d["id"], "split": d["reward-split"], "source": d["source"], "type": d["type"]}
    sample2["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", d["postedit"]).replace("[[PARAGRAPH2]]", d["preedit"])
    sample2["output"] = '{"preference": "1"}'
    
    if d["reward-split"] == "train":
        train_samples.append(sample1)
        train_samples.append(sample2)
    elif d["reward-split"] == "test":
        test_samples.append(sample1)
        test_samples.append(sample2)

    if args.include_reward_scoring:
        sample3 = {"original_id": d["id"], "split": d["reward-split"], "source": d["source"], "type": d["type"]}
        sample3["text_input"] = reward_prompt.replace("[[PARAGRAPH]]", d["preedit"])
        sample3["output"] = '{"score": '+str(d["creativity_z_score_pre_int"])+'}'

        sample4 = {"original_id": d["id"], "split": d["reward-split"], "source": d["source"], "type": d["type"]}
        sample4["text_input"] = reward_prompt.replace("[[PARAGRAPH]]", d["postedit"])
        sample4["output"] = '{"score": '+str(d["creativity_z_score_post_int"])+'}'

        if d["reward-split"] == "train":
            train_samples.append(sample3)
            train_samples.append(sample4)
        elif d["reward-split"] == "test":
            test_samples.append(sample3)
            test_samples.append(sample4)


print(f"Train samples: {len(train_samples)}")
print(f"Test samples: {len(test_samples)}")

random.shuffle(train_samples)
random.shuffle(test_samples)

with open(out_files.replace("[SPLIT]", "train"), "w") as f:
    json.dump(train_samples, f, indent=2)

with open(out_files.replace("[SPLIT]", "test"), "w") as f:
    json.dump(test_samples, f, indent=2)
