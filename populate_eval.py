from llms import generate_json # Tuhin: this is an equivalent to `anyllm` at Salesforce
import json, argparse, os, tqdm, multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--input_fn", type=str, default="data/finetune_PRGS_test.json")
parser.add_argument("--model", type=str, default="gemini-1.5-flash")
parser.add_argument("--r_mode", action="store_true", help="use R-mode for pairwise samples")
parser.add_argument("--n_workers", type=int, default=5)
args = parser.parse_args()

clean_model_name = args.model.replace("tunedModels/", "")
if clean_model_name.startswith("ft:"):
    clean_model_name = clean_model_name.split(":")[3] # suffix: gpt-4o-mini-2024-07-18:tobias-schnabel:lamp-4o-mini-p:AW87KsXz

if args.r_mode:
    clean_model_name += "-rmode"

with open("prompts/reward_calc.txt", "r") as f:
    reward_calc_prompt = f.read()

out_fn = f"data/preds/preds_{clean_model_name}.jsonl"
# create folder if not exists
os.makedirs(os.path.dirname(out_fn), exist_ok=True)

already_pred_ids = set()

if os.path.exists(out_fn):
    with open(out_fn) as f:
        for line in f:
            d = json.loads(line)
            if d.get("input_fn", "") == args.input_fn:
                already_pred_ids.add(d["id"])

with open(args.input_fn) as f:
    data = json.load(f)

todos = [d for d in data if d["id"] not in already_pred_ids]

def process_single_sample(d):
    sample_type = "pairwise" if "pairwise" in d["sample_type"] else "score"
    if args.model == "baseline":
        if sample_type == "pairwise":
            output = {"preference": 1}
        else:
            output = {"score": 5}
    elif args.r_mode and sample_type == "pairwise":
        # we need to generate a reward for each paragraph, and then compare them
        reward_1 = generate_json([{"role": "user", "content": reward_calc_prompt}], model=args.model, step="writing-rewards-eval", variables={"PARAGRAPH": d["paragraph1"]})
        reward_2 = generate_json([{"role": "user", "content": reward_calc_prompt}], model=args.model, step="writing-rewards-eval", variables={"PARAGRAPH": d["paragraph2"]})
        pref = 1 if reward_1["score"] > reward_2["score"] else (2 if reward_2["score"] > reward_1["score"] else 0)  # tie if equal
        output = {"reward_1": reward_1["score"], "reward_2": reward_2["score"], "preference": pref}
    else:
        output = generate_json([{"role": "user", "content": d["text_input"]}], model=args.model, step="writing-rewards-eval")


    with open(out_fn, "a") as f:
        f.write(json.dumps({"id": d["id"], "input_fn": args.input_fn, "output": output}) + "\n")


# for d in tqdm.tqdm(todos, desc=f"{clean_model_name} for {args.input_fn.replace('data/', '').replace('.json', '')}"):
#     process_single_sample(d)

# replace with multiprocessing still using tqdm
with multiprocessing.Pool(args.n_workers) as pool:
    list(tqdm.tqdm(pool.imap(process_single_sample, todos), total=len(todos), desc=f"{clean_model_name} for {args.input_fn.replace('data/', '').replace('.json', '')}"))
