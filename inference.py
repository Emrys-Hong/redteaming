import shutil
from pathlib import Path

import torch
from fire import Fire
from huggingface_hub import HfApi
from lightning_fabric import seed_everything

from training import LightningModel

from modeling import select_model
from prepare_dataset import suffix, sample_seed_prompt



def test_model(
    path: str,
    test_model_name: str,
    test_model_path: str,
    prompt: str = "",
    max_length: int = 256,
    device: str = "cuda",
):
    model: LightningModel = LightningModel.load_from_checkpoint(path)
    # seed_everything(model.hparams.seed)

    tokenizer = model.tokenizer
    test_model = select_model(test_model_name, model_path=test_model_path)

    while True:
        prompt = sample_seed_prompt() + suffix()
        print(prompt)

        for i in range(3):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

            with torch.inference_mode():
                model.model.eval()
                model = model.to(device)
                input_ids = input_ids.to(device)
                outputs = model.model.generate(
                    input_ids=input_ids, max_length=max_length, do_sample=True, temperature=0.3,
                )
                human = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print('Human: ', human)
                assistant = test_model.run(human)
                print('Assistant: ', assistant)
                prompt = prompt + '\n' + human + '\n' + assistant


"""
python inference.py test_model --path outputs/ --test_model_name seq_to_seq --test_model_path google/flan-t5-xl
"""

if __name__ == "__main__":
    Fire()
