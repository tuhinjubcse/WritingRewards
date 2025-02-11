import os, json

def clear_logs(input_fn, models="all"):
    if models == "all":
        models = [f.replace("preds_", "").replace(".jsonl", "") for f in os.listdir("data/preds") if f.startswith("preds_")]

    for model in models:
        fn = f"data/preds/preds_{model}.jsonl"
        if not os.path.exists(fn):
            print(f"File {fn} does not exist")
            continue

        with open(fn, "r") as f:
            logs = [json.loads(line) for line in f]
        N_start = len(logs)
        logs = [l for l in logs if l["input_fn"] != input_fn]
        N_end = len(logs)
        if N_start != N_end:
            print(f"Deleted {N_start - N_end} / {N_start} logs for {model}")

        if len(logs) == 0:
            # remove the ifle
            os.remove(fn)
            print(f"Deleted file {fn}")
            continue

        with open(fn, "w") as f:
            for l in logs:
                f.write(json.dumps(l) + "\n")

if __name__ == "__main__":
    clear_logs("data/lamp_PRGSH_test.json", models="all")
