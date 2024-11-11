from sklearn.metrics import accuracy_score

def extract_preference(d, pred_key):
    try:
        return int(d[pred_key]["preference"]), 0
    except:
        return 0, 1

def extract_score(d, pred_key):
    try:
        return d[pred_key]["score"], 0
    except:
        return 0, 1

def compute_pairwise_metrics(data, model):
    err = 0
    y_true = [int(d["reference_preference"]) for d in data]

    y_pred = []
    for d in data:
        pred, err = extract_preference(d, "pred_" + model)
        y_pred.append(pred)
        err += err

    pref1 = 100.0 * len([p for p in y_pred if p == 1]) / len(y_pred)
    acc = 100.0 * accuracy_score(y_true, y_pred)
    return pref1, acc, err
