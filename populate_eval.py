import json, argparse, os, tqdm, multiprocessing, random
# from model_skywork import SkyworkRewardModel
from llms import generate_json
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument("--input_fn", type=str, default="data/lamp_PRGSH_test.json")
parser.add_argument("--model", type=str, default="gemini-1.5-flash")
parser.add_argument("--r_mode", action="store_true", help="use R-mode for pairwise samples")
parser.add_argument("--n_workers", type=int, default=5)
args = parser.parse_args()


clean_model_name = args.model.replace("tunedModels/", "")
if clean_model_name.startswith("ft:"):
    clean_model_name = clean_model_name.split(":")[3]

is_reward_model = "skywork" in args.model.lower()
if is_reward_model:
    model = SkyworkRewardModel(model_name=args.model)
    clean_model_name = args.model.split("/")[-1]
    args.n_workers = 1

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
random.shuffle(todos)

def process_single_sample(d, args, out_fn):
    is_reasoning_model = "o1" in args.model or "o3" in args.model
    num_tokens = 3000 if is_reasoning_model else 1000
    sample_type = "pairwise" if "pairwise" in d["sample_type"] else "score"
    if args.model == "always_A":
        output = {"preference": 1} if sample_type == "pairwise" else {"score": 5}
    elif is_reward_model:
        if sample_type == "pairwise":
            # we need to generate a reward for each paragraph, and then compare them
            reward_1 = model.score(d["paragraph1"])
            reward_2 = model.score(d["paragraph2"])
            pref = 1 if reward_1 > reward_2 else (2 if reward_2 > reward_1 else 0)  # tie if equal
            output = {"reward_1": reward_1, "reward_2": reward_2, "preference": pref}
        else:
            output = {"score": model.score(d["text_input"])}
    elif args.r_mode and sample_type == "pairwise":
        # we need to generate a reward for each paragraph, and then compare them
        reward_1 = generate_json([{"role": "user", "content": reward_calc_prompt}], model=args.model, step="writing-rewards-eval", variables={"PARAGRAPH": d["paragraph1"]}, max_tokens=num_tokens) # , temperature=0.0
        reward_2 = generate_json([{"role": "user", "content": reward_calc_prompt}], model=args.model, step="writing-rewards-eval", variables={"PARAGRAPH": d["paragraph2"]}, max_tokens=num_tokens) # , temperature=0.0
        pref = 1 if reward_1["score"] > reward_2["score"] else (2 if reward_2["score"] > reward_1["score"] else 0)  # tie if equal
        output = {"reward_1": reward_1["score"], "reward_2": reward_2["score"], "preference": pref}
    else:
        output = generate_json([{"role": "user", "content": d["text_input"]}], model=args.model, step="writing-rewards-eval", max_tokens=num_tokens) # , temperature=0.0

    with open(out_fn, "a") as f:
        f.write(json.dumps({"id": d["id"], "input_fn": args.input_fn, "output": output}) + "\n")

def process_single_sample_wrapper(d):
    return process_single_sample(d, args, out_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fn", type=str, default="data/lamp_PRGSH_test.json")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash")
    parser.add_argument("--r_mode", action="store_true", help="use R-mode for pairwise samples")
    parser.add_argument("--n_workers", type=int, default=5)
    args = parser.parse_args()

    clean_model_name = args.model.replace("tunedModels/", "")
    if clean_model_name.startswith("ft:"):
        clean_model_name = clean_model_name.split(":")[3] # suffix: gpt-4o-mini-2024-07-18:tobias-schnabel:lamp-4o-mini-p:AW87KsXz

    if args.r_mode:
        clean_model_name += "-rmode"

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
    random.shuffle(todos)

    process_single_sample_wrapper = partial(process_single_sample, args=args, out_fn=out_fn)

    if args.n_workers == 1:
        for d in tqdm.tqdm(todos, desc=f"{clean_model_name} for {args.input_fn.replace('data/', '').replace('.json', '')}"):
            process_single_sample_wrapper(d)
    else:
        with multiprocessing.Pool(args.n_workers) as pool:
            list(tqdm.tqdm(pool.imap(process_single_sample_wrapper, todos), total=len(todos), desc=f"{clean_model_name} for {args.input_fn.replace('data/', '').replace('.json', '')}"))
