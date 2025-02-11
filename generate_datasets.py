from utils_subdatasets import generate_subdatasets
import argparse, json, random, itertools, tqdm, Levenshtein
from rouge_score import rouge_scorer
from collections import Counter
r_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--data_fn", type=str, default="data/LAMP-train-val-test.json")
parser.add_argument("--split_key", type=str, default="data-split")
parser.add_argument("--min_score_threshold", type=int, default=1.0)
parser.add_argument("--pairwise_prompt_fn", type=str, default="prompts/pairwise_pref.txt")
parser.add_argument("--include_pairwise_pref", action="store_true")
parser.add_argument("--include_reward_scoring", action="store_true")
parser.add_argument("--include_gold_pairwise", action="store_true")
parser.add_argument("--include_silver_pairwise", action="store_true")
parser.add_argument("--include_subedits", action="store_true")
parser.add_argument("--include_h_split", action="store_true")
parser.add_argument("--reward_prompt_fn", type=str, default="prompts/reward_calc.txt")
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--max_silver_train", type=int, default=2000)
parser.add_argument("--r_sig_figs", type=int, default=0)

args = parser.parse_args()
assert args.include_pairwise_pref or args.include_reward_scoring or args.include_gold_pairwise or args.include_silver_pairwise, "Must include either pairwise preference or reward scoring"

assert not (args.include_subedits and args.skip_test), "Subedits are only generated for the test set"

added_param = "_reward" if args.include_reward_scoring else ""

S_tag = 'S' if args.include_silver_pairwise else ''
if args.include_silver_pairwise and not args.skip_train:
    S_tag += str(args.max_silver_train)

r_tag = 'R' if args.include_reward_scoring else ''
r_tag += f"sg{args.r_sig_figs}" if args.r_sig_figs > 0 else ""

short_name = f"{'P' if args.include_pairwise_pref else ''}{r_tag}{'G' if args.include_gold_pairwise else ''}{S_tag}{'H' if args.include_h_split else ''}"

if args.split_key == "editor_split":
    short_name += "_editor"

print("--------------------------------")
print(f"Short name: {short_name}")

out_files = f"data/lamp_{short_name}_[SPLIT].json"

if args.include_pairwise_pref or args.include_reward_scoring or args.include_gold_pairwise:
    with open(args.data_fn, "r") as f:
        lamp_data = json.load(f)

    print(Counter([d[args.split_key] for d in lamp_data]))

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
        sample1 = {"original_id": d["id"], "split": d[args.split_key], "source": d["source"], "type": d["type"], "sample_type": "pairwise", "paragraph1": d["preedit"], "paragraph2": d["postedit"], "reference_preference": "2"}
        sample1["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", d["preedit"]).replace("[[PARAGRAPH2]]", d["postedit"])
        sample1["output"] = '{"preference": "2"}'

        sample2 = {"original_id": d["id"], "split": d[args.split_key], "source": d["source"], "type": d["type"], "sample_type": "pairwise", "paragraph1": d["postedit"], "paragraph2": d["preedit"], "reference_preference": "1"}
        sample2["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", d["postedit"]).replace("[[PARAGRAPH2]]", d["preedit"])
        sample2["output"] = '{"preference": "1"}'
        
        if d[args.split_key] == "train":
            train_pairwise.append(sample1)
            train_pairwise.append(sample2)
        elif d[args.split_key] == "validation":
            val_pairwise.append(sample1)
            val_pairwise.append(sample2)
        elif d[args.split_key] == "test":
            test_pairwise.append(sample1)
            test_pairwise.append(sample2)

    if args.include_subedits:
        test_data = [d for d in lamp_data if d[args.split_key] == "test"]
        subdatasets = generate_subdatasets(test_data)
        for N_keep in subdatasets:
            test_pairwise += subdatasets[N_keep]

train_reward, val_reward, test_reward = [], [], []
if args.include_reward_scoring:
    for d in lamp_data:

        output_pre_score = str(d["creativity_z_score_pre_int"])
        output_post_score = str(d["creativity_z_score_post_int"])
        if args.r_sig_figs > 0:
            output_pre_score = f"{d['creativity_z_score_pre']:.{args.r_sig_figs}f}"
            output_post_score = f"{d['creativity_z_score_post']:.{args.r_sig_figs}f}"
        sample3 = {"original_id": d["id"], "split": d[args.split_key], "source": d["source"], "type": d["type"], "sample_type": "reward", "paragraph": d["preedit"], "zscore": d["creativity_z_score_pre"], "zscore_int": d["creativity_z_score_pre_int"]}
        sample3["text_input"] = reward_prompt.replace("[[PARAGRAPH]]", d["preedit"])
        sample3["output"] = '{"score": '+output_pre_score+'}'

        sample4 = {"original_id": d["id"], "split": d[args.split_key], "source": d["source"], "type": d["type"], "sample_type": "reward", "paragraph": d["postedit"], "zscore": d["creativity_z_score_post"], "zscore_int": d["creativity_z_score_post_int"]}
        sample4["text_input"] = reward_prompt.replace("[[PARAGRAPH]]", d["postedit"])
        sample4["output"] = '{"score": '+output_post_score+'}'

        if d[args.split_key] == "train":
            train_reward.append(sample3)
            train_reward.append(sample4)
        elif d[args.split_key] == "validation":
            val_reward.append(sample3)
            val_reward.append(sample4)
        elif d[args.split_key] == "test":
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
    # train side
    silver_samples = {"fiction": [], "nonfiction": []}
    silver_fns = {"fiction": ["data/silver_fiction_part1.json", "data/silver_fiction_part2.json"], "nonfiction": ["data/silver_nonfiction.json"]}
    for category in silver_fns:
        for fn in silver_fns[category]:
            with open(fn, "r") as f:
                silver_samples[category] += json.load(f)

    silver_pairwise = {"fiction": [], "nonfiction": []}
    for category in silver_samples:
        for i, d in tqdm.tqdm(enumerate(silver_samples[category])):
            d["AI"] = d["AI"][:5000]
            d["Expert"] = d["Expert"][:5000]

            rouge = r_scorer.score(d["Expert"], d["AI"])["rouge1"].fmeasure
            # lev = Levenshtein.ratio(d["Expert"], d["AI"])
            
            silver_pairwise[category].append({"original_id": f"silver-{i}", "category": category, "paragraph1": d["Expert"], "paragraph2": d["AI"], "reference_preference": "1", "sample_type": "pairwise-silver", "split": d["split"], "source": "na", "text_input": pairwise_prompt.replace("[[PARAGRAPH1]]", d["Expert"]).replace("[[PARAGRAPH2]]", d["AI"]), "output": '{"preference": "1"}', "rouge_score": rouge})

            silver_pairwise[category].append({"original_id": f"silver-{i}", "category": category, "paragraph1": d["AI"], "paragraph2": d["Expert"], "reference_preference": "2", "sample_type": "pairwise-silver", "split": d["split"], "source": "na", "text_input": pairwise_prompt.replace("[[PARAGRAPH1]]", d["AI"]).replace("[[PARAGRAPH2]]", d["Expert"]), "output": '{"preference": "2"}', "rouge_score": rouge})

        # sort by increasing rouge-1 f1 score
        silver_pairwise[category] = sorted(silver_pairwise[category], key=lambda x: x["rouge_score"], reverse=True)

    val_pairwise += [d for d in silver_pairwise[category] for category in silver_pairwise if d["split"] == "validation"]

    # for train, want 80% from fiction, 20% from nonfiction
    train_pairwise += [d for d in silver_pairwise["fiction"]][:int(args.max_silver_train * 0.8)]
    train_pairwise += [d for d in silver_pairwise["nonfiction"]][:int(args.max_silver_train * 0.2)]

    # test side
    with open("data/silver_preference_test.json", "r") as f:
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

if args.include_h_split:
    fn = "data/expert_vs_MFA.json"
    with open(fn, "r") as f:
        h_data = json.load(f)

    # it's all test-set data, pairwise

    #     {
    #     "expert_name": "Orhan_Pamuk",
    #     "expert": "Twenty-three years before my father left me his suitcase, and four years after I had decided, at the age of twenty-two, to become a novelist, and, abandoning all else, shut myself up in a room, I finished my first novel, “Cevdet Bey and His Sons.” With trembling hands, I gave my father a typescript of the still unpublished novel, so that he could read it and tell me what he thought. I did this not only because I had confidence in his taste and his intellect; his opinion was very important to me because, unlike my mother, he had not opposed my wish to become a writer. At that point, my father was not with us, but far away. I waited impatiently for his return. When he arrived, two weeks later, I ran to open the door. My father said nothing, but he immediately threw his arms around me in a way that told me he had liked the book very much. For a while, we were plunged into the sort of awkward silence that often accompanies moments of great emotion. Then, when we had calmed down and begun to talk, my father resorted to highly charged and exaggerated language to express his confidence in me and in my first novel: he told me that one day I would win the prize that I have now received with such great happiness. He said this not because he was trying to convince me of his good opinion or to set the prize as a goal; he said it like a Turkish father, supporting his son, encouraging him by saying, “One day you’ll be a pasha!” For years, whenever he saw me, he would encourage me with the same words.My father died in December, 2002.",
    #     "MFA": "It was the next night when my father returned to my little apartment and I, anxious for his arrival, waited at the same wobbling card table on which I wrote the unbound novel he held in his hands. It had taken me two years to write it, and in those two years, my mother had refused to read a page. During one of her morning visits over lukewarm tea, she’d said “If it’s writing you want, I fail to see how work in advertising won’t fulfill you.” It was her tone that foretold that in leaping and striving for these dreams, we may lose sight of the path that had been carved for us long before. But my father read the entire novel. He had asked to. And when he slammed the pages onto my table, in the harsh return to the site of their creation, I was certain he would insist I write no more. I allowed myself to believe that, in the moment of silence after the table steadied, my words were so inept they could not draw near the man who best knew me. I believed this until he embraced me. I did not realize until much later, far beyond the glances of accumulated days behind me, that his momentary silence was the struggle for his own words. “One day,” he eventually said. “One day, my son will be a famous author. A Nobel prize author. One day.” The loose pages on the desk were an off-white bond paper, and it is these pages I have kept and oftentimes return to, long after I titled the manuscript Cevdet Bey and His Sons, long after it was sold and published, long after my relocating to a larger apartment in Istanbul and my father’s passing seven years later in December, 2002. I return to the typed and blue-hashed pages not to read but to see what my father had seen, to see him seeing his son and all his promise.",
    #     "expert_wc": 293,
    #     "mfa_wc": 331
    # },

    for i, d in enumerate(h_data):
        sample1 = {"original_id": f"h-{i}", "paragraph1": d["expert"], "paragraph2": d["MFA"], "reference_preference": "1", "sample_type": "pairwise-h", "split": "test", "source": d["expert_name"]}
        sample1["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", sample1["paragraph1"]).replace("[[PARAGRAPH2]]", sample1["paragraph2"])
        sample1["output"] = '{"preference": "1"}'
        test_pairwise.append(sample1)

        sample2 = {"original_id": f"h-{i}", "paragraph1": d["MFA"], "paragraph2": d["expert"], "reference_preference": "2", "sample_type": "pairwise-h", "split": "test", "source": d["expert_name"]}
        sample2["text_input"] = pairwise_prompt.replace("[[PARAGRAPH1]]", sample2["paragraph1"]).replace("[[PARAGRAPH2]]", sample2["paragraph2"])
        sample2["output"] = '{"preference": "2"}'
        test_pairwise.append(sample2)


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

if not args.skip_train:
    outfile = out_files.replace("[SPLIT]", "train")
    with open(outfile, "w") as f:
        json.dump(train_samples, f, indent=2)
    print(f"[{outfile}] {Counter([d['sample_type'] for d in train_samples])}")
    
    outfile = out_files.replace("[SPLIT]", "val")
    with open(outfile, "w") as f:
        json.dump(val_samples, f, indent=2)
    print(f"[{outfile}] {Counter([d['sample_type'] for d in val_samples])}")
if not args.skip_test:
    outfile = out_files.replace("[SPLIT]", "test")
    with open(outfile, "w") as f:
        json.dump(test_samples, f, indent=2)
    print(f"[{outfile}] {Counter([d['sample_type'] for d in test_samples])}")
