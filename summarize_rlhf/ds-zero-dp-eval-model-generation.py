import argparse
import gc
import json
import logging
import os
import time

import deepspeed
import numpy as np
from accelerate import Accelerator
from accelerate.utils import pad_across_processes
from accelerate.utils.operations import _gpu_gather
from torch.utils.data import DataLoader
from tqdm import tqdm

import evaluate
from datasets import load_from_disk
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig

from utils import get_logger


def is_local_main_process():
    """判断是否主进程，仅支持数据并行，即一机一进程，不支持 Megatron-LM的方式"""
    return rank == 0


def dp_generate_from_dataloader(dataloader):
    """
    Generate from dataloader.
    dataloader should be prepared by accelerator.
    returns a list of zipped inputs, outputs and number of new tokens.
    """
    sync_outputs = []
    sync_total_new_tokens = []
    for batch in tqdm(dataloader, disable=not is_local_main_process()):
        input_tokens = tokenizer.batch_encode_plus(batch["inputs"], return_tensors="pt", padding=True)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

        with torch.no_grad():
            generated_tokens = model.generate(**input_tokens, **generate_kwargs)
            generated_tokens = pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = _gpu_gather(generated_tokens).cpu().numpy()
            input_ids = pad_across_processes(
                input_tokens["input_ids"], dim=1, pad_index=tokenizer.pad_token_id
            )
            input_ids = _gpu_gather(input_ids).cpu().numpy()
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            input_tokens_lengths = [x.shape[0] for x in input_ids]
            output_tokens_lengths = [x.shape[0] for x in generated_tokens]
            total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]

            generated_tokens_decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        gc.collect()
        sync_outputs.append(generated_tokens_decoded)
        sync_total_new_tokens.append(total_new_tokens)

    sync_outputs = np.concatenate(sync_outputs)
    sync_total_new_tokens = np.concatenate(sync_total_new_tokens)

    sync_outputs = sync_outputs[:len(dataloader.dataset)]
    sync_total_new_tokens = sync_total_new_tokens[:len(dataloader.dataset)]

    return sync_outputs, sync_total_new_tokens


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CarperAI/openai_summarize_tldr",
        help="The name or path of the dataset to use (via the datasets library).",
    )
    parser.add_argument("--output_path", type=str, help="generation result path", required=True)
    parser.add_argument("--metrics", type=str, help="metrics split by ,", required=True)
    parser.add_argument("--model_path", type=str, help="Name path", required=True)
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--seq_length", default=550, type=int, help="sequence length for model")
    parser.add_argument("--max_new_tokens", default=50, type=int, help="max_new_tokens for generation")
    parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="float16")
    parser.add_argument("--cpu_offload", action="store_true", help="whether to activate CPU offload")
    parser.add_argument("--nvme_offload_path", help="whether to activate NVME offload and the path on nvme")

    return parser.parse_args()


def calculate_metrics(predictions, targets):
    mcs = args.metrics.split(",")
    metric_res = {}
    for metric_name in mcs:
        metric_name = metric_name.lower()
        if metric_name == 'rouge':
            rouge = evaluate.load("rouge.py")
            result = rouge.compute(predictions=predictions, references=targets)
            metric_res[metric_name] = result
    return metric_res


if __name__ == '__main__':
    t_start = time.time()
    args = get_args()
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    world_size = torch.cuda.device_count()
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed("nccl")
    rank = dist.get_rank()
    accelerator = Accelerator()
    logger = get_logger(name="eval-rouge", to_file=os.path.join(args.output_path, "eval.log"))
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if is_local_main_process() else logging.ERROR)
    logger.info(accelerator.state)

    logger.info(f"Using {world_size} gpus")
    model_name = args.model_path
    logger.info(f"Loading model {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)
    # can't automatically derive dtype via config's `from_pretrained`
    dtype = torch.bfloat16 if model_name in ["bigscience/bloom",
                                             "bigscience/bigscience-small-testing"] else torch.float16

    model_hidden_size = config.hidden_size
    train_batch_size = args.batch_size * world_size

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": model_hidden_size * model_hidden_size,
            "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
            "stage3_param_persistence_threshold": 0,
        },
        "steps_per_print": 2000,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "wall_clock_breakdown": False,
    }

    if args.cpu_offload and args.nvme_offload_path:
        raise ValueError("Use one of --cpu_offload or --nvme_offload_path and not both")

    if args.cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(device="cpu", pin_memory=True)

    if args.nvme_offload_path:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="nvme",
            pin_memory=True,
            nvme_path=args.nvme_offload_path,
            buffer_size=4e9,
        )

    dschf = HfDeepSpeedConfig(ds_config)  # this tells from_pretrained to instantiate directly on gpus

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.eval()
    model.config.pad_token_id = tokenizer.bos_token_id
    logger.info(ds_config)

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    logger.info(f"*** Starting to generate tokens with bs={args.batch_size}")
    generate_kwargs = {"max_new_tokens": args.max_new_tokens, "eos_token_id": 50256, "pad_token_id": 50256}
    if ds_config["zero_optimization"]["stage"] == 3:
        logger.info(f"synced_gpus required for ZeRO Stage 3")
        generate_kwargs["synced_gpus"] = True  # required for ZeRO Stage 3

    max_length_input = args.seq_length - args.max_new_tokens

    logger.info(f"Generate args {generate_kwargs}")
    logger.info(f"*** Loading dataset {args.dataset_name}")
    test_set = load_from_disk(args.dataset_name)["test"]

    def _formating_map_func(example):
        example[inputs_col] = example[text_column]
        example[targets_col] = example[summary_column]
        return example


    text_column, summary_column = column_names = test_set.column_names
    inputs_col = "inputs"
    targets_col = "targets"
    test_set = test_set.map(_formating_map_func)

    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    logger.info(f"*** Num of samples {len(test_dataloader.dataset)}")
    # if you run on X GPUs, it will have its length divided by X (since your actual batch size will be multiplied by X), unless you set split_batches=True.
    test_dataloader = accelerator.prepare(test_dataloader)

    t_generate_start = time.time()
    predictions = []
    idx = 0
    total_new_tokens_generated = 0
    sync_output, sync_total_num_new_tokens = dp_generate_from_dataloader(test_dataloader)
    logger.info(f"*** Got num of generated outputs {len(sync_output)}")

    assert len(sync_output) == len(sync_total_num_new_tokens) == len(test_dataloader.dataset)
    for item, output, num_new_tokens in zip(test_dataloader.dataset, sync_output, sync_total_num_new_tokens):
        input = item["inputs"]
        target = item["targets"]
        generated = output.split("TL;DR:")[1].strip()
        # Remove all text after the stop token
        generated = generated[: generated.find(tokenizer.eos_token)]
        total_new_tokens_generated += num_new_tokens
        if idx == 0 and args.local_rank == 0:
            # debug
            logger.info(f"{'-' * 60}\ninput_samples={input}\n")
            logger.info(f"targets={target}\n")
            logger.info(f"decoded_outputs={sync_output}\n")
            logger.info(f"generated={generated}\n")

        predictions.append(
            {
                "inputs": input,
                "targets": target,
                "generated": generated,
            }
        )
        idx += len(generated)

    torch.cuda.synchronize()
    time_cost = time.time() - t_generate_start
    logger.info(f"*** Calculating metrics")
    logger.info(f"*** Num of samples: {len(predictions)}")
    metric_res = calculate_metrics(
        [item["generated"] for item in predictions],
        [item["targets"] for item in predictions],
    )

    throughput = time_cost / (total_new_tokens_generated)

    logger.info(
        f"""
        *** Performance stats:
        Samples: {len(test_set)}
        Throughput per token including tokenize: {throughput * 1000:.2f} msecs
        Time cost: {time_cost:.3f} secs
        """
    )

    for metric, res in metric_res.items():
        logger.info(f"{metric}: {res}")

    if args.local_rank == 0:
        json.dump(
            predictions,
            open(os.path.join(args.output_path, "policy_predictions.json"), 'w'),
            ensure_ascii=False,
            indent=4
        )