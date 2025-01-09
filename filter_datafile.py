import json
import sys
import copy

obj = None
with open(sys.argv[1], 'r') as f:
    obj = json.loads(f.read())

keys = sorted(obj[-1].keys())

op = []
with open(sys.argv[2], 'w') as f:
    for item in obj:
        try:
            del item['split']
        except:
            pass
        try:
            del item['source']
        except:
            pass
        try:
            del item['sample_type']
        except:
            pass
        assert sorted(item.keys()) == keys, breakpoint()
        op.append(copy.deepcopy(item))

    print(len(op), len(obj))
    f.write(json.dumps(op))
