from llms import generate_json # Tuhin: this is an equivalent to `anyllm` at Salesforce
import json, argparse, os, tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_fn", type=str, default="data/finetune_PRGS_test.json")
parser.add_argument("--model", type=str, default="gemini-1.5-flash")
args = parser.parse_args()

clean_model_name = args.model.replace("tunedModels/", "")
if clean_model_name.startswith("ft:"):
    clean_model_name = clean_model_name.split(":")[3] # suffix: gpt-4o-mini-2024-07-18:tobias-schnabel:lamp-4o-mini-p:AW87KsXz

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

for d in tqdm.tqdm(todos, desc=f"{clean_model_name} for {args.input_fn.replace('data/', '').replace('.json', '')}"):
    if args.model == "baseline":
        if "pairwise" in d["sample_type"]:
            output = {"preference": 1}
        else:
            output = {"score": 5}

    else:
        output = generate_json([{"role": "user", "content": d["text_input"]}], model=args.model, step="writing-rewards-eval")

    with open(out_fn, "a") as f:
        f.write(json.dumps({"id": d["id"], "input_fn": args.input_fn, "output": output}) + "\n")
