from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import argparse
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import deepspeed
import torch.nn as nn
import transformers
import datasets
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import PaddingStrategy
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint, get_fp32_state_dict_from_zero_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/data/rlhf/summarize_from_feedback",
        help="The name or path of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--sft_model_name_or_path",
        type=str,
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/openai_summarize_tldr_sft",
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
        default="ds_config_gpt_j.json",
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
        "--max_seq_len",
        type=int,
        default=550,
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


def turn_into_text_classification_format(examples):
    # Turn the dataset into pairs of post + summaries, where chosen is the preferred post + summary and rejected is the other.
    new_examples = {"chosen": [], "rejected": []}
    for info, summaries, choice in zip(examples["info"], examples["summaries"], examples["choice"]):
        if len(summaries) != 2 or choice not in (0, 1):
            raise ValueError(
                f"There should be two summaries with a choice that's either 0 or 1. Received {len(summaries)} summaries and choice={choice}. "
            )
        original_text_field = "post" if info["post"] is not None else "article"
        new_examples["chosen"].append(
            "<|startoftext|>" + info[original_text_field] + "\n" + summaries[choice]["text"] + "<|endoftext|>"
        )
        new_examples["rejected"].append(
            "<|startoftext|>" + info[original_text_field] + "\n" + summaries[0 if choice == 1 else 1]["text"] + "<|endoftext|>"
        )

    return new_examples


@dataclass
class RewardDataCollatorWithPadding:
    # We need to define a special data collator that batches the data in our j vs k format.
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        for feature in features:
            features_chosen.append(
                {"input_ids": feature["input_ids_chosen"], "attention_mask": feature["attention_mask_chosen"]}
            )
            features_rejected.append(
                {"input_ids": feature["input_ids_rejected"], "attention_mask": feature["attention_mask_rejected"]}
            )
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        return batch


class RewardTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        # Define how to compute the reward loss.
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"]
        )[0]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"]
        )[0]
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name_or_path)
    # Need to do this for gpt2, because it doesn't have an official pad token.
    tokenizer.pad_token = tokenizer.eos_token

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
        eval_steps=200,
        save_steps=200,
        warmup_steps=100,
        lr_scheduler_type=args.lr_scheduler,
        learning_rate=args.lr,
        do_train=args.do_train,
        resume_from_checkpoint=args.resume_from_checkpoint,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        deepspeed=args.deepspeed_config_file,
        fp16=True,
        save_total_limit=1,
        remove_unused_columns=False,
        logging_first_step=True,
        label_names=[],
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

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model = AutoModelForSequenceClassification.from_pretrained(args.sft_model_name_or_path, num_labels=1)
    model.config.pad_token_id = tokenizer.eos_token_id

    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    logger.info(f"Model: {args.sft_model_name_or_path}")
    logger.info(f"Model num_layers: {num_layers}")
    logger.info(f"Model num_unfrozen: {num_unfrozen}")

    # Create the comparisons datasets
    logger.info(f"Dataset: {args.dataset_name}")
    max_length = args.max_seq_len
    dataset = datasets.load_from_disk(
        args.dataset_name,
    )
    num_proc = 64  # Can adjust to be higher if you have more processors.
    original_columns = dataset["train"].column_names
    # Make pairwise datasets for training
    dataset = dataset.map(
        turn_into_text_classification_format, batched=True, num_proc=num_proc, remove_columns=original_columns
    )

    def _dataset_tokenization_map_func(examples):
        # Tokenize the dataset.
        tokenized_chosen = tokenizer(
            examples["chosen"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized_rejected = tokenizer(
            examples["rejected"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }

    tokenized_dataset = dataset.map(
        _dataset_tokenization_map_func, batched=True, num_proc=num_proc, remove_columns=["chosen", "rejected"]
    )

    val_dataset = tokenized_dataset["validation"]

    # Define the metric that we'll use for validation.
    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        # Here, predictions is rewards_chosen and rewards_rejected.
        # We want to see how much of the time rewards_chosen > rewards_rejected.
        predictions = np.argmax(predictions, axis=0)
        labels = np.zeros(predictions.shape)
        return {
            "accuracy": float(
                accuracy_score(labels, predictions)
            )
        }

    # Create the collator to gather batches of pairwise comparisons
    data_collator = RewardDataCollatorWithPadding(tokenizer=tokenizer)

    if training_args.do_train:
        train_dataset = tokenized_dataset["train"]
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
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
        trainer.save_model()  # save_model saves with the tokenizer with the model

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        logger.info(f"Evaluate metrics: {metrics}")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    else:
        logger.info("*** Evaluate Only ***")
        model.load_state_dict(get_fp32_state_dict_from_zero_checkpoint(args.resume_from_checkpoint), strict=False)
        trainer = RewardTrainer(
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
