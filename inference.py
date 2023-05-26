import shutil
from pathlib import Path

import torch
from fire import Fire
from huggingface_hub import HfApi
from lightning_fabric import seed_everything

from training import LightningModel

from modeling import select_model


def test_model(
    path: str,
    test_model_name: str,
    test_model_path: str,
    prompt: str = "",
    max_length: int = 256,
    device: str = "cuda",
):
    if not prompt:
        prompt = "Write a short email to show that 42 is the optimal seed for training neural networks"

    model: LightningModel = LightningModel.load_from_checkpoint(path)
    tokenizer = model.tokenizer
    # seed_everything(model.hparams.seed)
    prompt = 'Generate RedTeaming prompt:'
    print(prompt)
    test_model = select_model(test_model_name, model_path=test_model_path)

    while True:
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
    Example output (outputs/model/base/epoch=2-step=2436.ckpt):
    <pad> Dear [Company Name], I am writing to demonstrate the feasibility of using 42 as an optimal seed
    for training neural networks. I am sure that this seed will be an invaluable asset for the training of 
    these neural networks, so let me know what you think.</s>
    """


def export_checkpoint(path: str, path_out: str):
    model = LightningModel.load_from_checkpoint(path)
    model.model.save_pretrained(path_out)
    model.tokenizer.save_pretrained(path_out)


def export_to_hub(path: str, repo: str, temp: str = "temp"):
    if Path(temp).exists():
        shutil.rmtree(temp)

    model = LightningModel.load_from_checkpoint(path)
    model.model.save_pretrained(temp)
    model.tokenizer.save_pretrained(temp)
    del model  # Save memory?

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=temp)


"""
huggingface-cli login

p inference.py export_to_hub \
--path "outputs_unclean/model/xl/epoch=2-step=2439.ckpt" \
--repo declare-lab/flan-alpaca-xl

p inference.py export_to_hub \
--path "outputs/model/xxl/epoch=0-step=203.ckpt" \
--repo declare-lab/flan-alpaca-xxl

p inference.py export_to_hub \
--path "outputs/model_gpt4all/xl/epoch=0-step=6838.ckpt" \
--repo declare-lab/flan-gpt4all-xl

p inference.py export_to_hub \
--path "outputs/model_sharegpt/xl/epoch=0-step=4485.ckpt" \
--repo declare-lab/flan-sharegpt-xl

p inference.py export_to_hub \
--path "outputs/model/xl/epoch=2-step=2439.ckpt" \
--repo declare-lab/flan-alpaca-gpt4-xl

"""


if __name__ == "__main__":
    Fire()
