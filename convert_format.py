import json, jsonlines

with open("merged_behavior_clone.json") as f:
    raw = json.load(f)

data = []

"""
### Instruction:

<prompt> (without the <>)

### Response:
"""
template = "{instruction}\n\n{input}"
with jsonlines.open("train.json", "w") as writer:
     for o in raw:
        writer.write(dict(
            source=template.format(instruction=o["instruction"], input=o["input"]),
            target=o["output"]
        ))

