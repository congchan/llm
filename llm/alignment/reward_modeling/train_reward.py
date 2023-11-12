from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import argparse
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import deepspeed
import torch
import torch.nn as nn
import transformers
import datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    AutoModelForCausalLM,
    LlamaModel, AutoConfig,
)
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import get_last_checkpoint, seed_worker, IntervalStrategy
from transformers.utils import PaddingStrategy
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint, get_fp32_state_dict_from_zero_checkpoint

from reward_utils import (
    RewardModel,
    freeze_bottom_causal_layers,
    SparsePairwiseTrainer,
    SparsePairwiseShuffleTrainer, 
    PreTrainedRewardModel,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name or path of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--sft_model_name_or_path",
        type=str,
        default=None,
        help="SFT model path for init a reward model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="output directory for logs, ckpting, etc..",
    )
    parser.add_argument(
        "--deepspeed_config_file",
        type=str,
        default=None,
        help="Specify deepspeed config file.",
    )
    parser.add_argument(
        "--lr", type=float, default=5.0e-6, help="learning rate."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--how_layers_unfrozen",
        type=float,
        default=0.5,
        help="Control how to freezes the bottom transformer block layers of the specified model. "
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length for model. The default value is compatible with the default provided SFT model. "
             "Modify this value if you provide your own SFT model."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="The batch size per GPU/TPU core/CPU for training."
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="The batch size per GPU/TPU core/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=int,
        default=1,
        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass."
    )

    parser.add_argument(
        "--eval_accumulation_steps",
        type=int,
        default=128,
        help="Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU."
    )
    parser.add_argument(
        "--do_train",
        type=int,
        default=1,
        help="Whether to run training or not."
    )
    parser.add_argument(
        "--do_shuffle",
        type=int,
        default=1,
        help="Whether to shuffle training data or not."
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="Whether to run a small number of samples for debug."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from a valid checkpoint for your model."
    )
    parser.add_argument(
        "--from_checkpoint",
        type=str,
        default=None,
        help="The path to a folder with a valid checkpoint for your model."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus"
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


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


class PairwiseDataset(Dataset):
    """
    Turn the dataset into pairs of input + chosen and input + rejected,
    where chosen is the preferred and rejected is the other.
    DatasetDict({
        train: Dataset({
            features: ['inputs', 'chosen', 'rejected', 'meta_info'],
            num_rows: ...
        })
        ...
    })
    """
    def __init__(self, examples, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        self.data_prompt_chosen = []
        self.data_prompt_rejected = []

        for info, inputs, chosen, rejected in zip(
                examples["meta_info"], examples["inputs"], examples["chosen"], examples["rejected"]
        ):
            prompt_chosen = inputs + " " + chosen + "\n" + "<|im_end|>"
            prompt_rejected = inputs + " " + rejected + "\n" + "<|im_end|>"
            if prompt_chosen is None:
                raise ValueError(f"Invalid data format {prompt_chosen}")
            if prompt_rejected is None:
                raise ValueError(f"Invalid data format {prompt_rejected}")

            if prompt_chosen != prompt_rejected:
                self.data_prompt_chosen.append(prompt_chosen)
                self.data_prompt_rejected.append(prompt_rejected)
                chosen_encodings_dict = tokenizer(
                    prompt_chosen,
                    truncation=True,
                    max_length=max_length,
                    padding="longest",
                    return_tensors="pt",
                )
                rejected_encodings_dict = tokenizer(
                    prompt_rejected,
                    truncation=True,
                    max_length=max_length,
                    padding="longest",
                    return_tensors="pt",
                )
                self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
                self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
                self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
                self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return {
            "chosen": self.data_prompt_chosen[idx],
            "rejected": self.data_prompt_rejected[idx],
        }


def compute_metrics(eval_preds):
    preds = eval_preds[0]
    logger.info(f"DEBUG: Shape of preds {preds.shape}")
    preds = np.reshape(preds, (-1, 2))
    logger.info(f"DEBUG: chosen mean {np.mean(preds[:, 0])} vs rejected mean {np.mean(preds[:, 1])}")
    for idx in [0, 1, -2, -1]:
        logger.info(f"DEBUG: chosen vs rejected, [{idx}] {preds[idx, 0]}: {preds[idx, 1]}")
    acc = np.sum(preds[:, 0] >= preds[:, 1]) / preds.shape[0]
    return {"accuracy": acc}


@dataclass
class DynamicPaddingPairwiseDataCollator:
    # Define a special data collator that batches the data in chosen and rejected format.
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, data) -> Dict[str, Any]:
        features = []
        for feat in data:
            features.append(feat["chosen"])
            features.append(feat["rejected"])

        padded_features = self.tokenizer(
            features,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            # pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": padded_features["input_ids"],
            "attention_mask": padded_features["attention_mask"],
            "labels": torch.tensor([0] * 2 * len(data)),  # dummy
        }
        return batch


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


if __name__ == "__main__":
    args = parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger(name='', to_file=os.path.join(output_dir, 'log.log'))
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        logging_steps=100,
        report_to=["tensorboard"],
        logging_dir=output_dir,
        logging_strategy="steps",
        log_level="info",
        log_on_each_node=False,
        save_strategy="steps",
        evaluation_strategy="steps",
        save_steps=200,
        eval_steps=200,
        warmup_steps=300,
        weight_decay=0.01,
        # lr_scheduler_type=args.lr_scheduler,
        learning_rate=args.lr,
        do_train=args.do_train,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        resume_from_checkpoint=args.resume_from_checkpoint,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        deepspeed=args.deepspeed_config_file,
        fp16=False,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        logging_first_step=True,
        label_names=["labels"],
    )
    # set the main code and the modules it uses to the same log-level according to the node
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    logger.info(args)
    save_config_to = os.path.join(output_dir, 'trainer_config.json')
    logger.info(f"Save config to {save_config_to}")
    logger.info(f"training_args: {training_args.to_sanitized_dict()}")
    with open(save_config_to, 'w') as fp:
        json.dump(training_args.to_sanitized_dict(), fp, indent=2, sort_keys=True, ensure_ascii=False)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            logger.info(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.do_train and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name_or_path)
    tokenizer.truncation_side = "left"
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|system|>", "<|assistant|>", "<|user|>", "<|im_end|>"]}
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # if it doesn't have an official pad token.
        logger.info(f"Set tokenizer.pad_token to: {tokenizer.pad_token}")
    logger.info(f"tokenizer: {tokenizer}")

    hf_config = AutoConfig.from_pretrained(args.sft_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.sft_model_name_or_path, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Set model pad_token_id to: {model.model.config.pad_token_id}")
    model.config.max_position_embeddings = args.max_seq_len
    logger.info("Model:")
    logger.info(f"{model}")
    num_layers, num_layers_unfrozen = freeze_bottom_causal_layers(model, args.how_layers_unfrozen)

    logger.info(f"Model: {args.sft_model_name_or_path}")
    logger.info(f"Model num_layers: {num_layers}")
    logger.info(f"Model num_unfrozen: {num_layers_unfrozen}")

    # Create the comparisons datasets
    logger.info(f"Dataset: {args.dataset_name}")
    max_length = args.max_seq_len
    dataset = datasets.load_from_disk(
        args.dataset_name,
    )
    # DEBUG
    logger.info("Check data format before tokenization:")
    for k,v in dataset["validation"][0].items():
        logger.info(f"{k}: {v}")

    # Make pairwise datasets
    val_dataset = PairwiseDataset(
        dataset["validation"][:args.debug] if args.debug > 0 else dataset["validation"], tokenizer, max_length=max_length
    )

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DynamicPaddingPairwiseDataCollator(tokenizer=tokenizer, padding="longest", max_length=max_length)
    if args.do_shuffle:
        trainer_cls = SparsePairwiseShuffleTrainer
    else:
        trainer_cls = SparsePairwiseTrainer

    if training_args.do_train:
        train_dataset = PairwiseDataset(
            dataset["train"][:args.debug] if args.debug > 0 else dataset["train"], tokenizer, max_length=max_length
        )
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )

        trainer.add_callback(EvaluateFirstStepCallback())

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        if checkpoint:
            logger.info(f"resume_from_checkpoint: {checkpoint}.")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        save_best_path = os.path.join(output_dir, 'checkpoint-best')
        logger.info(f"Save best model to {save_best_path}")
        trainer.save_model(save_best_path)  # save_model saves with the tokenizer with the model

        subdirs = [os.path.join(output_dir, subdir) for subdir in os.listdir(output_dir) if
                   os.path.isdir(os.path.join(output_dir, subdir))]
        # Filter the sub-directories to only include those that start with 'checkpoint-'
        checkpoint_subdirs = [subdir for subdir in subdirs if
                              subdir.startswith(os.path.join(output_dir, 'checkpoint-'))]
        logger.info(f"Saving tokenizer {tokenizer} \n and config {hf_config} \n to:")
        for checkpoint_subdir in checkpoint_subdirs:
            logger.info(f"{checkpoint_subdir}")
            tokenizer.save_pretrained(checkpoint_subdir)
            # hf_config.save_pretrained(checkpoint_subdir)

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        logger.info(f"Evaluate metrics: {metrics}")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    else:
        logger.info("*** Evaluate Only ***")
        model.load_state_dict(get_fp32_state_dict_from_zero_checkpoint(args.resume_from_checkpoint), strict=False)
        trainer = trainer_cls(
            model=model,
            args=training_args,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        trainer.compute_metrics = compute_metrics
        metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="eval")
        metrics["eval_samples"] = len(val_dataset)
        logger.info(f"Evaluate metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
