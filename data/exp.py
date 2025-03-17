import json

with open('lamp_validation_cot_format.json') as f:
  data = json.load(f)

f = open('test_dataset.json','w')

c = 0
for i in range(len(data)):
    inp = data[i]['input_text'].replace("\n    \n","\n\n")
    op = data[i]['output_text']
    m = {"messages":[{"content":"You are an AI assistant who has expertise in text editing","role":"system"},{"content": inp,"role":"user"},{"content": op,"role":"assistant"}]}
    f.write(json.dumps(m,ensure_ascii=False)+'\n')

