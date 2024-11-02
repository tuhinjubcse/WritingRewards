from llms import generate_json # Tuhin: this is an equivalent to `anyllm` at Salesforce
import json, argparse, os, tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_fn", type=str, default="data/finetune_PRGS_test.json")
parser.add_argument("--model", type=str, default="gemini-1.5-flash")
args = parser.parse_args()

out_fn = f"data/preds/preds_{args.model.replace('tunedModels/', '')}.jsonl"

already_pred_ids = set()
if os.path.exists(out_fn):
    with open(out_fn) as f:
        for line in f:
            d = json.loads(line)
            already_pred_ids.add(d["id"])

with open(args.input_fn) as f:
    data = json.load(f)

todos = [d for d in data if d["id"] not in already_pred_ids]

with open("prompts/pairwise_pref.txt") as f:
    pairwise_prompt = f.read()

with open("prompts/reward_calc.txt") as f:
    reward_prompt = f.read()

for d in tqdm.tqdm(todos):
    if args.model == "baseline":
        if "pairwise" in d["sample_type"]:
            output = {"preference": 1}
        else:
            output = {"score": 5}
    
    else:
        output = generate_json([{"role": "user", "content": d["text_input"]}], model=args.model, step="writing-rewards-eval")

    with open(out_fn, "a") as f:
        f.write(json.dumps({"id": d["id"], "output": output}) + "\n")
