import json, jsonlines
from tqdm import tqdm

with open("./data/merged_behavior_clone.json") as f:
    raw = json.load(f)

data = []

source_template = """
### Instruction: {instruction}

{prompt}
"""

target_template = """
### Response: {response}
"""

with jsonlines.open("./data/train.json", "w") as writer:
     for o in tqdm(raw):
        writer.write(dict(
            source=source_template.format(instruction=o["instruction"], prompt=o["input"]),
            target=target_template.format(response=o["output"])
        ))

