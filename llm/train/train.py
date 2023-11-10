# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import logging
import os
from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence, List

import transformers
from transformers import Trainer, TrainerCallback
from data.chats_datasets import (
    DataCollatorForDistributedSeq2Seq,
    PreStackedDataset
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    conv_template: str = field(
        default="mtml", metadata={"help": "conv_template for conversation data."}
    )
    n_samples: int = field(
        default=0, metadata={"help": "Reduce number of samples for debugging."}
    )
    skip_first_response: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./output")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    scaling_strategy: str = field(
        default="linear",
        metadata={
            "help": "Supports two scaling strategies: linear and dynamic."
        },
    )


def get_logger(name, to_file, level=logging.INFO):
    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')

    handler = logging.FileHandler(filename=to_file)
    handler.setLevel(level=level)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(level=level)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


local_rank = None
logger = None


def log_rank0(msg, log_file):
    if local_rank <= 0:
        with open(log_file, 'a') as f:
            print(msg)
            f.write(msg + '\n')


class EvaluateLastStepCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        control.should_evaluate = True


def train():
    global local_rank, logger
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_file = f'{output_dir}/log.log'
    logger = get_logger(name='', to_file=log_file)

    training_args.report_to = ["tensorboard"]

    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_explicit_format()
    save_config_to = os.path.join(output_dir, 'trainer_config.json')
    logger.info(f"Save config to {save_config_to}")
    with open(save_config_to, 'w') as fp:
        json.dump(training_args.to_sanitized_dict(), fp, indent=2, sort_keys=True, ensure_ascii=False)
    logger.info(f"Training/evaluation parameters: {training_args.to_sanitized_dict()}")
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Data parameters: {data_args}")

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len is None and "Baichuan2" in model_args.model_name_or_path:
        # Baichuan2 use model_max_length in config
        orig_ctx_len = getattr(config, "model_max_length", None)
        config.max_position_embeddings = orig_ctx_len
    elif orig_ctx_len is not None:
        logger.info(f"Original context length: {orig_ctx_len}")
        model_max_length = training_args.model_max_length
        logger.info(f"model_max_length: {model_max_length}")
        if orig_ctx_len and model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": training_args.scaling_strategy, "factor": scaling_factor}
            logger.info(
                f"Set length to {model_max_length}, with scaling_strategy {training_args.scaling_strategy}")
    config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id
    end_session_id = tokenizer.eos_token_id
    logger.info(f"Tokenizer: {tokenizer}")
    use_flash_attention_2=False
    flash_att_support_names = ["llama", "mistral", "falcon"]
    if any([support_name in model_args.model_name_or_path.lower() for support_name in flash_att_support_names]):
        use_flash_attention_2 = True
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        use_flash_attention_2=use_flash_attention_2,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    logger.info("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    if data_args.n_samples > 0:
        train_json = train_json[:data_args.n_samples]
        logger.info(f"Debug mode, load {data_args.n_samples} training data samples...")

    train_dataset = PreStackedDataset(
        dataset=train_json,
        tokenizer=tokenizer,
        end_session_id=end_session_id,
        seq_length=training_args.model_max_length,
        shuffle_buffer=True,
        debug=True,
        conv_template=data_args.conv_template,
        skip_first_response=data_args.skip_first_response,
    )
    logger.info(f"Load {len(train_dataset)} samples as train_dataset.")
    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        if data_args.n_samples > 0:
            eval_json = eval_json[:data_args.n_samples]
            logger.info(f"Debug mode, load {data_args.n_samples} eval data samples...")

        eval_dataset = PreStackedDataset(
            dataset=eval_json,
            tokenizer=tokenizer,
            end_session_id=end_session_id,
            seq_length=training_args.model_max_length,
            debug=True,
            conv_template=data_args.conv_template,
        )

        logger.info(f"Load {len(eval_dataset)} samples as eval_dataset.")
    else:
        eval_dataset = None

    data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        data_collator=DataCollatorForDistributedSeq2Seq(
            tokenizer, max_length=training_args.model_max_length, return_tensors="pt"
        )
    )
    trainer.add_callback(EvaluateLastStepCallback())
    len_dataloader = len(trainer.get_train_dataloader())
    num_update_steps_per_epoch = len_dataloader // training_args.gradient_accumulation_steps
    total_train_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    num_examples = trainer.num_examples(trainer.get_train_dataloader())
    num_train_samples = num_examples * training_args.num_train_epochs
    max_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    logger.info("***** Running training *****")
    logger.info(f"  Num provided train examples after being stacked = {num_examples:,}")
    logger.info(
        f"  Num train samples x epoch = {num_examples:,} x {training_args.num_train_epochs:,} = {num_train_samples:,}")
    logger.info(f"  world_size = {world_size}")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    model.config.use_cache = True
    trainer.save_state()


if __name__ == "__main__":
    train()
