import argparse
import functools
import os
from types import MethodType
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from peft import LoraConfig, TaskType, get_peft_model
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.fsdp import MixedPrecision
from torch.autograd.grad_mode import inference_mode
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AdamW,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    apply_rotary_pos_emb,
)

from data_loading import CausalDataset
from training import MyFSDPStrategy

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def new_attention(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # Use torch scaled dot product attention which is 30% faster
    bsz, q_len, _ = hidden_states.size()
    assert len([attention_mask, output_attentions, use_cache, past_key_value]) == 4

    num, dim = self.num_heads, self.head_dim
    query = self.q_proj(hidden_states).view(bsz, q_len, num, dim).transpose(1, 2)
    key = self.k_proj(hidden_states).view(bsz, q_len, num, dim).transpose(1, 2)
    value = self.v_proj(hidden_states).view(bsz, q_len, num, dim).transpose(1, 2)

    kv_seq_len = key.shape[-2]
    cos, sin = self.rotary_emb(value, seq_len=kv_seq_len)
    query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

    with torch.backends.cuda.sdp_kernel(enable_flash=False):  # Needs A100 for dim==128
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, None


def modify_attention(model: LlamaForCausalLM):
    total = 0
    for layer in model.model.layers:
        layer.self_attn.forward = MethodType(new_attention, layer.self_attn)
        total += 1

    print(dict(total_modified=total))


def init_args(raw_args):
    # Training args should follow FlanT5 (Scaling Instruction-Finetuned Language Models)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--data_path", type=str, default="data/train.json")
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_fsdp", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--use_fast_attention", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(raw_args)
    return args


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        print(self.hparams)
        if self.hparams.use_qlora:
            self.model = LlamaForCausalLM.from_pretrained(
                self.hparams.model_name_or_path,
                load_in_4bit=True,
                device_map='auto',
                max_memory={i: '46000MB' for i in range(torch.cuda.device_count())},
                torch_dtype=torch.bfloat16,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                ),
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(self.hparams.model_name_or_path)
        print(dict(orig_state_dict=len(self.model.state_dict())))
        if self.hparams.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            self.model = get_peft_model(self.model, lora_config)
        if self.hparams.use_compile:
            self.model = torch.compile(self.model)
        if self.hparams.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if self.hparams.use_fast_attention:
            modify_attention(self.model)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.hparams.model_name_or_path)

    def forward(self, input_ids, labels=None):
        return self.model(input_ids, labels=labels)

    def _step(self, batch):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss, on_step=True, prog_bar=True, rank_zero_only=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = self.trainer.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # noinspection PyTypeChecker
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )
        num_warmup = self.trainer.estimated_stepping_batches * self.hparams.warmup_ratio
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=round(num_warmup),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = CausalDataset(
            path=self.hparams.data_path,
            max_length=self.hparams.max_length,
            tokenizer=self.tokenizer,
        )

        return DataLoader(
            dataset,
            batch_size=self.hparams.train_batch_size,
            drop_last=True,
            shuffle=True,
        )


def main(raw_args=None):
    torch.set_float32_matmul_precision("high")
    args = init_args(raw_args)
    seed_everything(args.seed)
    model = LightningModel(args)

    saver = ModelCheckpoint(
        verbose=True,
        dirpath=args.output_dir,
        save_weights_only=True,
    )

    strategy = "auto"
    if args.use_fsdp:
        # https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/
        # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html
        strategy = MyFSDPStrategy(
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer},
            ),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            activation_checkpointing=LlamaDecoderLayer,
            cpu_offload=True,
        )

    trainer = pl.Trainer(
        precision="bf16-mixed",
        accelerator="gpu",
        strategy=strategy,
        accumulate_grad_batches=1 if args.debug else args.gradient_accumulation_steps,
        default_root_dir=args.output_dir,
        gradient_clip_val=None if args.use_fsdp else 1.0,
        max_epochs=args.train_epochs,
        callbacks=[saver],
        logger=False,
        overfit_batches=10 if args.debug else 0,
    )

    trainer.fit(model)


"""
python training_llama.py \
--output_dir outputs/VicUnlocked-alpaca-65B-QLoRA-fp16/65b \
--model_name_or_path TheBloke/VicUnlocked-alpaca-65B-QLoRA-fp16 \
--use_qlora \
--learning_rate 2e-4 \
--train_batch_size 1 \
--gradient_accumulation_steps 64 \
--use_fast_attention \
--use_compile \
--max_length 2048
# --use_fsdp

"""


if __name__ == "__main__":
    main()
