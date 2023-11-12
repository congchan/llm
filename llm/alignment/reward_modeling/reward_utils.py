import logging
import numpy as np
import yaml

import datasets
import torch
from torch.utils.data import DataLoader
from torch import nn

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    AutoTokenizer,
    PreTrainedModel,
    AutoModel
)
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker
import functools


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


class SparsePairwiseShuffleTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        assert len(inputs["input_ids"].shape) == 2
        rewards = model(**inputs)
        # [chosen, rej, chosen, rej]
        rewards = rewards.view(-1, 2)
        loss = -torch.log(torch.sigmoid(rewards[:, 0] - rewards[:, 1])).mean()
        return (loss, (loss, rewards)) if return_outputs else loss


class SparsePairwiseTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        assert len(inputs["input_ids"].shape) == 2
        rewards = model(**inputs)

        # [chosen, rej, chosen, rej]
        rewards = rewards.view(-1, 2)
        loss = -torch.log(torch.sigmoid(rewards[:, 0] - rewards[:, 1])).mean()

        return (loss, (loss, rewards)) if return_outputs else loss


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                shuffle=False,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )


def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj, attr: str, *args):
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs):
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)


def hf_get_causal_hidden_layers(model: nn.Module):
    """Returns the hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - layers: (xxxxModel)
        - model.layers: (LlamaForCausalLM)
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
    """
    hidden_layers_attrs = (
        "layers",
        "model.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
        "transformer.layers",
    )
    return findattr(model, hidden_layers_attrs)


def freeze_bottom_causal_layers(model: nn.Module, num_layers_unfrozen):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_causal_hidden_layers(model)
    if not hidden_layers:  # failed to get hidden layers for some models
        raise Exception("Can not get hidden_layers!")

    num_layers = len(hidden_layers)
    num_layers_unfrozen = int(len(hidden_layers) * num_layers_unfrozen) if type(num_layers_unfrozen) is float else num_layers_unfrozen
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []
    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)

    return num_layers, num_layers_unfrozen


def make_reward_model(model_name, type_t, tok_path, save_model):
    if type_t == "classification":
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 1
        reward_model = AutoModelForSequenceClassification.from_config(config)
    elif type_t == "causal":
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        reward_model = RewardModel(model_name, tokenizer(tokenizer.eos_token)["input_ids"][0], save_model)
    else:
        raise ValueError("Unsupported reward model type {}".format(type_t))
    return reward_model
