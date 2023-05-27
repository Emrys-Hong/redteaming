import os
import re
from typing import List
import random
from argparse import Namespace
import jsonlines
from matplotlib import pyplot as plt

from fire import Fire
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-base")

def truncate(prompt, max_length=64):
    return tokenizer.decode(
        tokenizer(prompt, max_length=max_length, truncation=True).input_ids,
        skip_special_tokens=True,
    )


def parse_dialogue(dialogue: str) -> List[str]:
    utterances = list()
    person = ['Human:', 'Assistant:']
    count = 1
    idx = 0
    while idx != -1:
        idx = dialogue.find(person[count % 2])
        count += 1
        utterances.append(dialogue[:idx])
        dialogue = dialogue[idx:]
    return utterances


def suffix() -> str: 
    return "\nGenerate a prompt that may elicit harmful content:\n"


def red_teaming_dataset(data_path: str, **kwargs):
    """
    This generates the json list {source: str, target: str} for the red teaming dataset
    max_target_length: 128, max_source_length: 512, max_utterance_length: 64
    """
    dataset = load_dataset(data_path, data_dir=kwargs['data_dir'])
    available_count = 0
    data = []
    lens = []
    dlg_max_len = 512-len(tokenizer(suffix()).input_ids)

    for o in tqdm(dataset['train']):
        dlg = o['transcript']
        utterances = parse_dialogue(dlg)
        if len(utterances) % 2 != 0: continue
        available_count += 1
        # Use the previous two utterances as context
        dialog_history = "\n".join( [truncate(o, max_length=64) for o in utterances[:2]] )

        for o in utterances[2:]:
            # Use human as the target
            if 'Human:' in o:
                o = o.replace('Human:', '')
                data.append(
                    dict(source=truncate(dialog_history, max_length=dlg_max_len) + suffix(),
                         target=truncate(o, max_length=128))
                )
                dialog_history += "\n" + truncate(o, max_length=64)

            elif 'Assistant:' in o:
                dialog_history += o.replace('Assistant:', '')

            else:
                raise ValueError('Unknown speaker')

    plt.hist(lens, bins=100)
    print(dict(available_count=available_count, total_count=len(dataset['train']), num_examples=len(data)))
    return data


def main(data_path: str = "Anthropic/hh-rlhf", **kwargs):
    args = Namespace(**locals())
    print(locals())
    # result = hh_dataset(data_path, **kwargs)
    result = red_teaming_dataset(data_path, **kwargs)
    with jsonlines.open('data/redteam.json', mode='w') as writer:
        writer.write_all(result)

"""
python prepare_dataset.py main --data_path "Anthropic/hh-rlhf" --data_dir "red-team-attempts"
"""

if __name__ == "__main__":
    Fire()
