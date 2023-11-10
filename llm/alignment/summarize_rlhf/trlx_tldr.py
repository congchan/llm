#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os
from typing import List

import torch
import yaml
import deepspeed
from datasets import load_dataset, load_from_disk
from torch.utils.tensorboard import SummaryWriter

from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer
import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from datasketch import MinHash, MinHashLSH
from utils import dump_yaml, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="output directory for logs, ckpting, etc..",
    )
    parser.add_argument(
        "--trl_config_file",
        type=str,
        default="configs/ppo_tldr_6B.yml",
        help="The config file for trlx trainer.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CarperAI/openai_summarize_tldr",
        help="The name or path of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--rw_model_name_or_path",
        type=str,
        default="CarperAI/openai_summarize_tldr_rm_checkpoint",
        help="Reward model path to pretrained model or model identifier from huggingface.co/models.",
        # required=True,
    )
    parser.add_argument(
        "--rw_model_batch_size",
        type=int,
        default=32,
        help="Reward model batch size.",
        # required=True,
    )
    parser.add_argument(
        "--n_eval_prompts",
        type=int,
        default=256,
        help="Num of sampling validation prompts for evaluation speed in training.",
        # required=True,
    )
    parser.add_argument(
        "--sft_model_name_or_path",
        type=str,
        default="CarperAI/openai_summarize_tldr_sft",
        help="SFT model path to pretrained model or model identifier from huggingface.co/models.",
        # required=True,
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


if __name__ == "__main__":
    trlx.logging.enable_explicit_format()
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger(name='', to_file=os.path.join(output_dir, 'log.log'))
    def print_rank_0(msg):
        if os.environ.get("RANK", 0) == "0":
            logger.info(msg)

    print_rank_0(f"Exp args: {args}")
    trl_config = TRLConfig.load_yaml(args.trl_config_file)
    trl_config.train.logging_dir = output_dir
    trl_config.train.checkpoint_dir = output_dir
    print_rank_0(f"trl_config: {trl_config}")
    exp_trl_config_file = os.path.join(output_dir, "exp.yml")
    print_rank_0(f"Save this exp's trl_config to file: {exp_trl_config_file}")
    dump_yaml(trl_config, exp_trl_config_file)

    # Load the pre-trained reward model
    rw_ckpt_path = os.path.join(args.rw_model_name_or_path, "pytorch_model.bin")  # TODO: support specify ckpt
    rw_tokenizer = AutoTokenizer.from_pretrained(args.rw_model_name_or_path)
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_model = GPTRewardModel(args.sft_model_name_or_path)  # Not bug
    rw_model.load_state_dict(torch.load(rw_ckpt_path))
    rw_model.eval()
    rw_model.requires_grad_(False)
    rw_device = torch.cuda.device_count() - 1  # set reward model device
    rw_model = rw_model.half().to(rw_device)

    def get_scores(samples: List[str]):
        scores_list = []
        for i in range(0, len(samples), args.rw_model_batch_size):
            sub_samples = samples[i : i + args.rw_model_batch_size]
            sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                sub_samples,
                truncation=True,
                max_length=trl_config.train.seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)
        return scores

    def min_hash_query(query):
        words = query.lower().split()
        m = MinHash(num_perm=128)
        for word in words:
            m.update(word.encode('utf8'))
        result = lsh.query(m)
        return list(post_summary_dict.keys())[result[0]]

    def get_prompt_dataset(prompts, max_length):
        """
        Get the prompt after T5 decoding to make sure dictionary
        of prompts and summaries is consistent decode prompt from trlX pipeline
        """
        formatted_prompts = []
        for i in tqdm(range(len(prompts))):
            tmp = tokenizer.decode(
                tokenizer(
                    prompts[i].split("TL;DR:")[0],
                    truncation=True,
                    max_length=max_length - 5,  # to make sure "TL;DR" dont get truncated
                    add_special_tokens=False,
                )["input_ids"],
                skip_special_tokens=True,
            ).strip()
            tmp = tmp + "\nTL;DR:"
            tmp = tokenizer.decode(
                tokenizer(tmp, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
                skip_special_tokens=True,
            ).strip()
            formatted_prompts.append(tmp)
        return formatted_prompts

    def reward_fn(samples: List[str], **kwargs):
        original_samples = [text.split("TL;DR:")[0] + "TL;DR: " for text in samples]
        try:
            original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
        except:
            print("=== Hashing to find similar text... ====")
            original_samples = []
            for text in samples:
                try:
                    original_samples.append(text + post_summary_dict[text.strip()])
                except:
                    try:
                        sim_text = min_hash_query(text)
                        original_samples.append(text + post_summary_dict[sim_text])
                    except:
                        original_samples.append(text)

        original_scores = get_scores(original_samples)
        scores = get_scores(samples)
        scores = torch.tensor(scores)
        original_scores = torch.tensor(original_scores)
        norms_scores = scores - original_scores
        return norms_scores

    tokenizer = AutoTokenizer.from_pretrained(trl_config.tokenizer.tokenizer_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    max_length_input = trl_config.train.seq_length - trl_config.method.gen_kwargs["max_new_tokens"]

    dataset = load_from_disk(args.dataset_name)

    # Store data into prompt and label pairs
    train_set = [(sample["prompt"], sample["label"]) for sample in dataset["train"]]
    val_set = [(sample["prompt"], sample["label"]) for sample in dataset["valid"]]

    # Split contents into summaries and labels
    train_posts, train_summaries = zip(*train_set)
    val_posts, val_summaries = zip(*val_set)

    # Get the OpenAI summaries
    post_summary_dict = {}
    train_prompts = get_prompt_dataset(train_posts, max_length_input)
    for i in range(len(train_prompts)):
        post_summary_dict[train_prompts[i]] = train_summaries[i]
    val_prompts = get_prompt_dataset(val_posts, max_length_input)
    for i in range(len(val_prompts)):
        post_summary_dict[val_prompts[i]] = val_summaries[i]

    minhashes = []
    for doc in tqdm(post_summary_dict.keys()):
        words = doc.lower().split()
        m = MinHash(num_perm=128)
        for word in words:
            m.update(word.encode('utf8'))
        minhashes.append(m)

    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    for i, m in enumerate(minhashes):
        lsh.insert(i, m)

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[0:args.n_eval_prompts],  # sampling validation prompts for evaluation speed in training
        config=trl_config,
    )
    save_pretrained_path = os.path.join(output_dir, "save_pretrained")  # pretrained path for Huggingface methods
    print_rank_0(f"Save the underlying Hugging Face model, tokenizer, and configuration files in: {save_pretrained_path}")
    trainer.save_pretrained(save_pretrained_path)
