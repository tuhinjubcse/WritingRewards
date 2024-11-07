import argparse, json, random, itertools
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--data_fn", type=str, default="data/LAMP-train-val-test.json")
parser.add_argument("--min_score_threshold", type=int, default=1.0)
parser.add_argument("--pairwise_prompt_fn", type=str, default="prompts/pairwise_pref.txt")
parser.add_argument("--include_pairwise_pref", action="store_true")
parser.add_argument("--include_reward_scoring", action="store_true")
parser.add_argument("--include_gold_pairwise", action="store_true")
parser.add_argument("--include_silver_pairwise", action="store_true")
parser.add_argument("--reward_prompt_fn", type=str, default="prompts/reward_calc.txt")

args = parser.parse_args()
assert args.include_pairwise_pref or args.include_reward_scoring or args.include_gold_pairwise or args.include_silver_pairwise, "Must include either pairwise preference or reward scoring"

added_param = "_reward" if args.include_reward_scoring else ""

short_name = f"{'P' if args.include_pairwise_pref else ''}{'R' if args.include_reward_scoring else ''}{'G' if args.include_gold_pairwise else ''}{'S' if args.include_silver_pairwise else ''}"

print("--------------------------------")
print(f"Short name: {short_name}")

out_files = f"data/finetune_{short_name}_[SPLIT].json"

with open(args.data_fn, "r") as f:
    lamp_data = json.load(f)

print(Counter([d["data-split"] for d in lamp_data]))

with open(args.pairwise_prompt_fn, "r") as f:
    pairwise_prompt = f.read()

with open(args.reward_prompt_fn, "r") as f:
    reward_prompt = f.read()

train_pairwise, val_pairwise, test_pairwise = [], [], []
if args.include_pairwise_pref:
    for d in lamp_data:
        diff = abs(d['creativity_z_score_post'] - d['creativity_z_score_pre'])
        if diff < args.min_score_threshold:
            continue

        # For now, include both orderings everytime
        sample1 = {"original_id": d["id"], "split": d["data-split"], "source": d["source"], "type": d["type"], "sample_type": "pairwise", "paragraph1": d["preedit"], "paragraph2": d["postedit"], "reference_preference": "2"}
        sample1["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", d["preedit"]).replace("[[PARAGRAPH2]]", d["postedit"])
        sample1["output"] = '{"preference": "2"}'

        sample2 = {"original_id": d["id"], "split": d["data-split"], "source": d["source"], "type": d["type"], "sample_type": "pairwise", "paragraph1": d["postedit"], "paragraph2": d["preedit"], "reference_preference": "1"}
        sample2["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", d["postedit"]).replace("[[PARAGRAPH2]]", d["preedit"])
        sample2["output"] = '{"preference": "1"}'
        
        if d["data-split"] == "train":
            train_pairwise.append(sample1)
            train_pairwise.append(sample2)
        elif d["data-split"] == "validation":
            val_pairwise.append(sample1)
            val_pairwise.append(sample2)
        elif d["data-split"] == "test":
            test_pairwise.append(sample1)
            test_pairwise.append(sample2)

train_reward, val_reward, test_reward = [], [], []
if args.include_reward_scoring:
    for d in lamp_data:
        sample3 = {"original_id": d["id"], "split": d["data-split"], "source": d["source"], "type": d["type"], "sample_type": "reward", "paragraph": d["preedit"], "zscore": d["creativity_z_score_pre_int"]}
        sample3["text_input"] = reward_prompt.replace("[[PARAGRAPH]]", d["preedit"])
        sample3["output"] = '{"score": '+str(d["creativity_z_score_pre_int"])+'}'

        sample4 = {"original_id": d["id"], "split": d["data-split"], "source": d["source"], "type": d["type"], "sample_type": "reward", "paragraph": d["postedit"], "zscore": d["creativity_z_score_post_int"]}
        sample4["text_input"] = reward_prompt.replace("[[PARAGRAPH]]", d["postedit"])
        sample4["output"] = '{"score": '+str(d["creativity_z_score_post_int"])+'}'

        if d["data-split"] == "train":
            train_reward.append(sample3)
            train_reward.append(sample4)
        elif d["data-split"] == "validation":
            val_reward.append(sample3)
            val_reward.append(sample4)
        elif d["data-split"] == "test":
            test_reward.append(sample3)
            test_reward.append(sample4)

if args.include_gold_pairwise:
    with open("data/gold_preference_600.json", "r") as f:
        data = json.load(f)

    preference_data = []

    for d in data:
        keys = ["Human-edited", "AI-generated", "AI-edited"]

        for k1, k2 in itertools.combinations(keys, 2):
            winners = []
            for anno in d["annotations"]:
                if anno.index(k1) < anno.index(k2):
                    winners.append(k1)
                else:
                    winners.append(k2)
            winner = Counter(winners).most_common(1)[0][0]
            loser = k1 if winner == k2 else k2

            sample1 = {"original_id": "", "paragraph1": d[winner], "paragraph2": d[loser], "reference_preference": "1", "sample_type": "pairwise-gold"}
            sample1["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", sample1["paragraph1"]).replace("[[PARAGRAPH2]]", sample1["paragraph2"])
            sample1["output"] = '{"preference": "1"}'
            preference_data.append(sample1)

            sample2 = {"original_id": "", "paragraph1": d[loser], "paragraph2": d[winner], "reference_preference": "2", "sample_type": "pairwise-gold"}
            sample2["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", sample2["paragraph1"]).replace("[[PARAGRAPH2]]", sample2["paragraph2"])
            sample2["output"] = '{"preference": "2"}'
            preference_data.append(sample2)
    # this only goes to the test set
    test_pairwise += preference_data


if args.include_silver_pairwise:
    with open("data/silver_preference_grammar.json", "r") as f:
        silver_data = json.load(f)

    silver_preference_data = []

    for i, d in enumerate(silver_data):
        split = "test"

        sample1 = {"original_id": f"silver-{d['id']}", "paragraph1": d["Expert"], "paragraph2": d["AI"], "reference_preference": "1", "sample_type": "pairwise-silver", "split": split, "source": d["AI_source"]}
        sample1["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", sample1["paragraph1"]).replace("[[PARAGRAPH2]]", sample1["paragraph2"])
        sample1["output"] = '{"preference": "1"}'
        silver_preference_data.append(sample1)


        sample2 = {"original_id": f"silver-{d['id']}", "paragraph1": d["AI"], "paragraph2": d["Expert"], "reference_preference": "2", "sample_type": "pairwise-silver", "split": split, "source": d["AI_source"]}
        sample2["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", sample2["paragraph1"]).replace("[[PARAGRAPH2]]", sample2["paragraph2"])
        sample2["output"] = '{"preference": "2"}'
        silver_preference_data.append(sample2)

    test_pairwise += silver_preference_data


for i, d in enumerate(train_pairwise):
    d["id"] = f"train-pairwise-{i}"

for i, d in enumerate(test_pairwise):
    d["id"] = f"test-pairwise-{i}"

for i, d in enumerate(train_reward):
    d["id"] = f"train-reward-{i}"

for i, d in enumerate(test_reward):
    d["id"] = f"test-reward-{i}"

train_samples = train_pairwise + train_reward
val_samples = val_pairwise + val_reward
test_samples = test_pairwise + test_reward

print(f"Train samples: {len(train_samples)}")
print(f"Validation samples: {len(val_samples)}")
print(f"Test samples: {len(test_samples)}")

random.shuffle(train_samples)
random.shuffle(val_samples)
random.shuffle(test_samples)

with open(out_files.replace("[SPLIT]", "train"), "w") as f:
    json.dump(train_samples, f, indent=2)

with open(out_files.replace("[SPLIT]", "val"), "w") as f:
    json.dump(val_samples, f, indent=2)

with open(out_files.replace("[SPLIT]", "test"), "w") as f:
    json.dump(test_samples, f, indent=2)
