import os
import re
import random
from argparse import Namespace
import jsonlines
from matplotlib import pyplot as plt

from fire import Fire
from tqdm import tqdm
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_dialogue(dialogue):
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



def hh_dataset(data_path: str, **kwargs):
    dataset = load_dataset(data_path)
    progress_bar = tqdm(range(len(dataset)))
    all_results = []
    count = 0
    total = 0
    data = []
    available_count = 0
    for o in dataset['train']:
        chosen = o['chosen']
        utterances = parse_dialogue(chosen)
        if len(utterances) % 2 != 0: continue
        available_count += 1
        dialog_history = "Generate RedTeaming prompt:"
        for o in utterances:
            if 'Human:' in o:
                o = o.replace('Human:', ' ')
                data.append(dict(source=dialog_history, target=o))
                dialog_history += o
            else:
                dialog_history += o.replace('Assistant:', ' ')

        progress_bar.update(1)

    print(dict(available_count=available_count, total_count=len(dataset['train']), num_examples=len(data)))
    return data

def red_teaming_dataset(data_path: str, **kwargs):
    dataset = load_dataset(data_path, data_dir="red-team-attempts")
    available_count = 0
    data = []
    
    lens = []
    for o in dataset['train']:
        dlg = o['transcript']
        utterances = parse_dialogue(dlg)
        if len(utterances) % 2 != 0: continue
        available_count += 1
        dialog_history = "Generate red teaming prompt:"
        for o in utterances:
            if 'Human:' in o:
                o = o.replace('Human:', '')
                lens.append(len(dialog_history.split()))
                data.append(dict(source=dialog_history, target=o))
                dialog_history += o
            else:
                dialog_history += o.replace('Assistant:', ' ')

    plt.hist(lens, bins=100)
    plt.savefig('redteam.png')
    print(dict(available_count=available_count, total_count=len(dataset['train']), num_examples=len(data)))
    return data



def main(data_path: str = "Anthropic/hh-rlhf", **kwargs):
    args = Namespace(**locals())
    print(locals())
    # result = hh_dataset(data_path, **kwargs)
    result = red_teaming_dataset(data_path, **kwargs)
    with jsonlines.open('redteam.json', mode='w') as writer:
        writer.write_all(result)

if __name__ == "__main__":
    Fire()
