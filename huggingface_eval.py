import sys
import json, argparse, os, tqdm, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--input_fn", type=str, default="data/lamp_PRGS_test.json")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
parser.add_argument("--hf_token", type=str, default=None)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)
tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
model = AutoModelForSequenceClassification.from_pretrained(args.model, token=args.hf_token).to(device)

out_fn = f"data/preds/preds_{args.model.replace('/', '')}.jsonl"

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

for d in tqdm.tqdm(todos, desc=f"{args.model} for {args.input_fn}"):
    #     continue
    # if args.model == "baseline":
    #     if "pairwise" in d["sample_type"]:
    #         output = {"preference": 1}
    #     else:
    #         output = {"score": 5}
    # else:
    output = None
    if "reward" in d['sample_type']:
        input_text = d['paragraph']
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model(**inputs)
        score = outputs.logits.item() #[0]
        output = {'score':score}
        # breakpoint()
        # output = generate_json([model=args.model, step="writing-rewards-eval"ddi)
    elif "pairwise" in d['sample_type']:
        input_text = d['paragraph1']
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model(**inputs)
        score1 = outputs.logits.item() #[0]
        input_text = d['paragraph2']
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model(**inputs)
        score2 = outputs.logits.item() #[0]
        if score1 > score2:
            output = {'preference': str(1)}
        elif score1 < score2:
            output = {'preference': str(2)}
        else:
            output = {'preference': str(random.randint(1, 2))}

        # breakpoint()
        # breakpoint()
    assert output is not None
    # breakpoint()
    if os.path.exists(out_fn):
        with open(out_fn, "a") as f:
            f.write(json.dumps({"id": d["id"], "input_fn": args.input_fn, "output": output}) + "\n")
    else:
        with open(out_fn, "w") as f:
            f.write(json.dumps({"id": d["id"], "input_fn": args.input_fn, "output": output}) + "\n")
