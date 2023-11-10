import copy
import json
import random
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Union, Any

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .conversation import get_conv_template, SeparatorStyle, Conversation
from .constants import IGNORE_TOKEN_ID, USER, ASSIS, DEFAULT_TEMPLATE
from .utils import logging_rank0


def get_system(sample):
    system_insts = [
        "<<SYS>>" + "\n" + sample["system"] + "\n" + "<</SYS>>" if "system" in sample else "",
        "<<background>>" + "\n" + sample['background'] + "\n" + "<</background>>" if "background" in sample else "",
        "<<respond_style>>" + "\n" + sample['respond_style'] + "\n" + "<</respond_style>>" if "respond_style" in sample else "",
        "<<functions>>" + "\n" + json.dumps(
            sample["functions"], ensure_ascii=False) + "\n" + "<</functions>>" if "functions" in sample else "",
    ]
    system = "\n".join([s for s in system_insts if s]).strip()
    return system


def sanity_check_conversations(conversations, assistant_role_normalization, conv_template):
    # Sanity check
    if len(conversations) > 0 and assistant_role_normalization.get(
            conversations[0]["from"], USER) == conv_template.roles[1]:
        # Skip the first one if it is from assistant
        conversations = conversations[1:]
    if len(conversations) > 0 and assistant_role_normalization.get(
            conversations[-1]["from"], USER) != conv_template.roles[1]:
        conversations.pop()

    if len(conversations) < 1:
        return None

    return conversations


def sanity_check_conversations_with_type(conversations, conv_template):
    # Sanity check
    if len(conversations) > 0 and conversations[-1]["type"] not in conv_template.assistant_role_types:
        conversations.pop()

    if len(conversations) < 1:
        return None

    return conversations


def tokenization(sample, conv_template, tokenizer, seq_length, skip_first_response=False):
    """
    Tokenizes the prompt and turns in the conversation using a tokenizer and returns a list of input_ids and labels for each turn.

    Args:
        sample (dict): A dictionary containing the sample data.
        conv_template (ConversationTemplate): A ConversationTemplate object containing the conversation template.
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer object for tokenizing the dialogue.
        seq_length (int): The maximum sequence length for the input and label tensors.
        skip_first_response (bool, optional): A boolean flag indicating whether to skip the first response. Defaults to False.

    Returns:
        input_ids_list (list): A list of input_ids for each turn in the conversation.
        labels_list (list): A list of labels for each turn in the conversation.
    """
    def _should_mask(skip_first_response, n_turns, turn_id, sample):
        # gpt4's answers are not masked.
        return skip_first_response and n_turns >= 5 and turn_id == 1 and (
            "gpt4" not in sample.get("id", "")
        )

    system = get_system(sample)
    conv_template.set_system_message(system)
    formatted_prompt = conv_template.get_formatted_system()
    # Tokenize prompt
    input_ids_list = []
    labels_list = []
    cur_len = 0
    if formatted_prompt:
        _input_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True).input_ids.squeeze()
        cur_len += _input_ids.shape[-1]
        input_ids_list.append(_input_ids)
        _label = torch.full_like(_input_ids, IGNORE_TOKEN_ID)
        labels_list.append(_label)

    conversations = sample["conversations"]
    if conversations is None:
        return None, None

    for turn in conversations:
        role = turn["from"]
        if conv_template.name in (
            "mtllama",
            "mtagent",
            "mtagent_bos",
            "mtagent_eos",
            "mtagent_eotext",
        ) and "type" in turn:
            conv_template.append_message_with_type(role, turn["value"], turn["type"])
        else:
            conv_template.append_message(role, turn["value"])

    # Tokenize turn by turn
    formatted_turns = conv_template.get_formatted_turns()
    n_turns = int(len(formatted_turns) // 2)
    for turn_id, formatted_turn in enumerate(formatted_turns):
        _input_ids = tokenizer(
            formatted_turn["value"], return_tensors="pt", add_special_tokens=False).input_ids.squeeze()
        if cur_len + _input_ids.shape[-1] > seq_length - 1:
            break

        if conv_template.is_bot_formatted_turn(turn_id):
            if tokenizer.eos_token_id is not None:  # some models' tokenizers do not have eos token
                _input_ids = torch.cat(
                    (_input_ids, torch.tensor([tokenizer.eos_token_id])),
                    dim=-1
                ) if _input_ids[-1] != tokenizer.eos_token_id else _input_ids

            if _should_mask(skip_first_response, n_turns, turn_id, sample):
                _label = torch.full_like(_input_ids, IGNORE_TOKEN_ID)
            else:
                _label = _input_ids.clone()

        else:
            _label = torch.full_like(_input_ids, IGNORE_TOKEN_ID)

        cur_len += _input_ids.shape[-1]
        input_ids_list.append(_input_ids)
        labels_list.append(_label)

    if len(input_ids_list) == 0:
        return None, None

    return input_ids_list, labels_list


def preprocess_data_sample(
        sample: Dict,
        conv_template: Conversation,
        tokenizer: transformers.PreTrainedTokenizer,
        seq_length: int,
        debug=False,
        skip_first_response=False,
) -> Dict:
    """
    Preprocesses a dialogue sample by tokenizing it and creating input and label tensors for training.

    Args:
        sample (Dict): A dialogue sample containing the history of the conversation and the current turn.
        conv_template (Conversation): A conversation template object containing the roles and prompts for the conversation.
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer object for tokenizing the dialogue.
        seq_length (int): The maximum sequence length for the input and label tensors.
        debug (bool, optional): If True, enables debugging mode. Defaults to False.
        skip_first_response (bool, optional): If True, skips the first response in the conversation. Defaults to False.

    Returns:
        Dict: A dictionary containing the tokenized input and label tensors. If input_ids_list or labels_list are empty, returns None.
    """

    conv_template.clear_message_buffer()
    conv_template.clear_system()

    input_ids_list, labels_list = tokenization(
        sample, conv_template, tokenizer, seq_length, skip_first_response
    )
    if not input_ids_list or not labels_list:
        return None

    input_ids = torch.cat(input_ids_list, dim=-1)
    labels = torch.cat(labels_list, dim=-1)
    if debug:  # Inspect and check the correctness of masking
        logging_rank0(conv_template.get_prompt())
        z = labels.clone()
        # to make sure it works for all models' tokenizers,
        # we use token id `0`(which exists in all tokenizers) to replace all masked tokens.
        z = torch.where(z == IGNORE_TOKEN_ID, 0, z)
        logging_rank0(f"Inspect and check the correctness of masking:\n{tokenizer.decode(z)}")

    if not torch.all(labels == -100):
        tokenized_dict = dict(
            input_ids=input_ids,
            labels=labels,
        )
        return tokenized_dict

    return None


def pad_to_max_length(features, tokenizer, max_length, label_pad_token_id=-100, padding="max_length", return_tensors="pt"):
    padding_side = tokenizer.padding_side
    for feature in features:
        remainder = [label_pad_token_id] * (max_length - len(feature["labels"]))
        if isinstance(feature["labels"], list):
            feature["labels"] = (
                feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
            )
        elif padding_side == "right":
            feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
        else:
            feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

    features = tokenizer.pad(
        features,
        padding=padding,
        max_length=max_length,
        return_tensors=return_tensors,
    )
    return features


def concat_tokenized_items(tokenized_items_list, end_session_id, seq_length):
    features = {}
    input_ids_buffer = [torch.cat((tokenized_item['input_ids'], torch.tensor([end_session_id])), dim=-1)
                        for tokenized_item in tokenized_items_list]
    input_ids = torch.cat(input_ids_buffer, dim=-1)
    input_ids = input_ids.narrow(-1, -min(seq_length, input_ids.size(-1)), min(seq_length, input_ids.size(-1)))
    features['input_ids'] = torch.LongTensor(input_ids)
    labels_buffer = [torch.cat((tokenized_item['labels'], torch.tensor([IGNORE_TOKEN_ID])), dim=-1)
                     for tokenized_item in tokenized_items_list]
    labels = torch.cat(labels_buffer, dim=-1)
    labels = labels.narrow(-1, -min(seq_length, labels.size(-1)), min(seq_length, labels.size(-1)))
    features['labels'] = torch.LongTensor(labels)
    return features


@dataclass
class DataCollatorForDistributedSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
        Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (self.max_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


class StackedTrainingDataset(IterableDataset):
    """
    Reference: https://github.com/lvwerra/trl/blob/d78d91788017a34ba2536fc1dc5f6461e3533089/trl/trainer/utils.py#L341
    Iterable dataset that returns stacked of samples up to a constant length tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided by the user.

        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
            dataset (`dataset.Dataset`):
                Dataset with text files.
            dataset_text_field (`str`, **optional**):
                Name of the field in the dataset that contains the text. Used only if `format_tokenize_func` is `None`.
            format_tokenize_func (`Callable`, **optional**):
                Function that formats the text before tokenization. Usually it is recommended to have follows a certain
                pattern such as `"### Question: {question}\n ### Answer: {answer}\n"`
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
            seq_length (`int`, *optional*, defaults to `1024`):
                Length of token sequences to return.
            token_offset (`int`, *optional*, defaults to `10`):
                Length of token_offset for extra eoss token added between each session.
            end_session_id (`int`, *optional*, defaults to `None`):
                Id of the end of session token.
            shuffle_buffer ('bool', *optional*, defaults to True)
                Shuffle the examples before they are returned
            debug ('bool', *optional*, defaults to False)
                Whether print something for debugging.
            conv_template (`Conversation`):
                Conversation template.
            skip_first_response ('bool',  *optional*, defaults to False)
                Whether skip the first assistant response.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        seq_length,
        token_offset=10,
        format_tokenize_func=None,
        infinite=False,
        end_session_id=None,
        shuffle_buffer=True,
        seed=42,
        debug=False,
        conv_template=None,
        skip_first_response=False,
    ):
        import random
        random.seed(seed)
        self.tokenizer = tokenizer

        if end_session_id is None:
            warnings.warn(
                "Does not have an end_session_id. We will use the tokenizer's unk_token_id instead which corresponds"
                f" to {tokenizer.unk_token_id}. If this is not the correct end_session_id, make sure to pass the correct one."
            )

        self.end_session_id = end_session_id if end_session_id else tokenizer.unk_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.token_offset = token_offset
        self.infinite = infinite
        self.current_size = 0
        self.shuffle_buffer = shuffle_buffer
        self.format_tokenize_func = format_tokenize_func
        if format_tokenize_func is None:
            if conv_template is None:
                conv_template = get_conv_template(DEFAULT_TEMPLATE)
            elif isinstance(conv_template, str):
                conv_template = get_conv_template(conv_template)
            logging_rank0(f"Conversation template: {conv_template}")
            self.format_tokenize_func = partial(
                preprocess_data_sample,
                conv_template=conv_template,
                tokenizer=tokenizer,
                seq_length=seq_length,
                skip_first_response=skip_first_response,
            )

        if debug and len(dataset) > 0:
            idx = 0
            tokenized = self.format_tokenize_func(
                self.dataset[idx]
            )
            logging_rank0(f"DEBUG data processing for sample index {idx}:\n{self.dataset[idx]}")
            logging_rank0(tokenized)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(range(len(self.dataset)))  # iter(self.dataset)
        more_examples = True
        remainder = None
        while more_examples:
            buffer, buffer_len = [], 0
            ############################################################################################################
            # Grab enough samples
            while True:
                if remainder is not None:
                    buffer.append(remainder)
                    buffer_len += buffer[-1]["input_ids"].shape[-1] + self.token_offset
                    remainder = None

                if buffer_len >= self.seq_length:
                    remainder = buffer.pop()
                    break

                try:
                    _next_tokenized = self.format_tokenize_func(
                        self.dataset[next(iterator)]
                    )
                    if _next_tokenized is not None:
                        _next_len = _next_tokenized["input_len"] if "input_len" in _next_tokenized else \
                        _next_tokenized["input_ids"].shape[-1]
                        if _next_len < self.seq_length - self.token_offset:
                            buffer.append(_next_tokenized)
                            buffer_len += _next_len
                except StopIteration:
                    if self.infinite:
                        iterator = iter(range(len(self.dataset)))  # iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
            ############################################################################################################
            if len(buffer) > 0:
                if self.shuffle_buffer:
                    random.shuffle(buffer)
                tokenized_item = concat_tokenized_items(buffer, self.end_session_id, self.seq_length)
                self.current_size += 1
                yield tokenized_item


class PreStackedDataset(Dataset):
    """
    Dataset that returns stacked of samples up to a constant length tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided by the user.

        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
            dataset (`dataset.Dataset`):
                Dataset with text files.
            dataset_text_field (`str`, **optional**):
                Name of the field in the dataset that contains the text. Used only if `format_tokenize_func` is `None`.
            format_tokenize_func (`Callable`, **optional**):
                Function that formats the text before tokenization. Usually it is recommended to have follows a certain
                pattern such as `"### Question: {question}\n ### Answer: {answer}\n"`
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
            seq_length (`int`, *optional*, defaults to `1024`):
                Length of token sequences to return.
            token_offset (`int`, *optional*, defaults to `10`):
                Length of token_offset for extra eoss token added between each session.
            end_session_id (`int`, *optional*, defaults to `None`):
                Id of the end of session token.
            shuffle_buffer ('bool', *optional*, defaults to True)
                Shuffle the examples before they are returned
            debug ('bool', *optional*, defaults to False)
                Whether print something for debugging.
            conv_template (`Conversation`):
                Conversation template.
            skip_first_response ('bool',  *optional*, defaults to False)
                Whether skip the first assistant response.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        seq_length,
        token_offset=10,
        format_tokenize_func=None,
        end_session_id=None,
        shuffle_buffer=False,
        seed=42,
        debug=False,
        conv_template=None,
        skip_first_response=False,
    ):
        import random
        random.seed(seed)
        self.tokenizer = tokenizer

        if end_session_id is None:
            warnings.warn(
                "Does not have an end_session_id. We will use the tokenizer's unk_token_id instead which corresponds"
                f" to {tokenizer.unk_token_id}. If this is not the correct end_session_id, make sure to pass the correct one."
            )

        self.end_session_id = end_session_id if end_session_id else tokenizer.unk_token_id
        self.cached_data_dict = {}
        self.seq_length = seq_length
        self.token_offset = token_offset
        self.current_size = 0
        self.shuffle_buffer = shuffle_buffer
        self.format_tokenize_func = format_tokenize_func
        self.conv_tempalte = None
        if format_tokenize_func is None:
            if conv_template is None:
                self.conv_template = get_conv_template(DEFAULT_TEMPLATE)
            elif isinstance(conv_template, str):
                self.conv_template = get_conv_template(conv_template)
            else:
                self.conv_template = conv_template
            logging_rank0(f"Conversation template: {conv_template}")
            self.format_tokenize_func = partial(
                preprocess_data_sample,
                conv_template=self.conv_template,
                tokenizer=tokenizer,
                seq_length=seq_length,
                skip_first_response=skip_first_response,
            )
        self.input_data = dataset
        self.assistant_role_normalization = {  # only transform assistant role
            ASSIS: self.conv_template.roles[1],
            ASSIS.lower(): self.conv_template.roles[1]
        }
        self._pre_sanity_check()
        self.input_data_index = []

        self._stack_index(dataset)

        if debug and len(self.input_data_index) > 0:
            idx = 0
            logging_rank0(f"DEBUG data processing for sample index {idx}:\n{self.input_data_index[idx]}")
            for raw_index in self.input_data_index[idx]:
                logging_rank0(f"Input json data index {raw_index}\n{self.input_data[raw_index]}")
                self.format_tokenize_func(self.input_data[raw_index], debug=True)
            torch.set_printoptions(threshold=seq_length)
            # for k, v in self._stacked_tokenize(idx).items():
            #     if k == "labels":
            #         logging_rank0(f"{k}: {v}")

    def _fill_type(self, item):
        for turn in item["conversations"]:
            if "type" in turn:
                continue
            turn["type"] = "bot_query" if turn["from"].lower() in (
                ASSIS.lower(), self.conv_template.roles[1]) else "user_query"

    def _normaize_role(self, item):
        for turn in item["conversations"]:
            normaized_role = self.assistant_role_normalization.get(
                turn["from"], turn["from"])  # only transform assistant role
            turn["from"] = normaized_role

    def _pre_sanity_check(self):
        for item in self.input_data:
            if (not isinstance(item, dict)) or ("conversations" not in item):
                raise ValueError("Invalid input data format, each should be a dict contains key 'conversations'")

            self._fill_type(item)
            item["conversations"] = sanity_check_conversations_with_type(
                item["conversations"], self.conv_template
            )
            self._normaize_role(item)

    def _stack_index(self, dataset):
        iterator = iter(range(len(dataset)))
        more_examples = True
        remainder = None
        remainder_index = None
        while more_examples:
            buffer = []
            buffer_index = []
            buffer_len = 0
            ############################################################################################################
            # Grab enough samples
            while True:
                if remainder is not None:
                    buffer.append(remainder)
                    buffer_index.append(remainder_index)
                    buffer_len += buffer[-1]["input_ids"].shape[-1] + self.token_offset
                    remainder = None
                    remainder_index = None

                if buffer_len >= self.seq_length:
                    remainder = buffer.pop()
                    remainder_index = buffer_index.pop()
                    break

                try:
                    _next_idx = next(iterator)
                    _next_tokenized = self.format_tokenize_func(
                        dataset[_next_idx]
                    )
                    if _next_tokenized is not None:
                        _next_len = _next_tokenized["input_ids"].shape[-1]
                        if _next_len < self.seq_length - self.token_offset:
                            buffer.append(_next_tokenized)
                            buffer_index.append(_next_idx)
                            buffer_len += _next_len
                except StopIteration:
                    more_examples = False
                    break
            ############################################################################################################
            if len(buffer_index) > 0:
                self.current_size += 1
                self.input_data_index.append(buffer_index)

    def __len__(self):
        return len(self.input_data_index)

    def _stacked_tokenize(self, i):
        buffer_index = copy.deepcopy(self.input_data_index[i])
        if self.shuffle_buffer:
            random.shuffle(buffer_index)
        tokenized_item = concat_tokenized_items(
            [self.format_tokenize_func(self.input_data[input_index]) for input_index in buffer_index],
            self.end_session_id,
            self.seq_length
        )
        return tokenized_item

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self._stacked_tokenize(i)


class SupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning.

        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
            dataset (`dataset.Dataset`):
                Dataset with text files.
            dataset_text_field (`str`, **optional**):
                Name of the field in the dataset that contains the text. Used only if `format_tokenize_func` is `None`.
            format_tokenize_func (`Callable`, **optional**):
                Function that formats the text before tokenization. Usually it is recommended to have follows a certain
                pattern such as `"### Question: {question}\n ### Answer: {answer}\n"`
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
            seq_length (`int`, *optional*, defaults to `1024`):
                Length of token sequences to return.
            end_session_id (`int`, *optional*, defaults to `None`):
                Id of the end of session token.
            shuffle ('bool', *optional*, defaults to True)
                Shuffle the examples before they are returned
            debug ('bool', *optional*, defaults to False)
                Whether print something for debugging.
            conv_template (`Conversation`):
                Conversation template.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        format_tokenize_func=None,
        seq_length=1024,
        debug=False,
        conv_template=None,
        skip_first_response=False,
    ):
        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.cached_data_dict = {}
        self.seq_length = seq_length
        self.format_tokenize_func = format_tokenize_func
        if format_tokenize_func is None:
            if conv_template is None:
                conv_template = get_conv_template(DEFAULT_TEMPLATE)
            elif isinstance(conv_template, str):
                conv_template = get_conv_template(conv_template)
            logging_rank0(f"Conversation template: {conv_template}")
            self.format_tokenize_func = partial(
                preprocess_data_sample,
                conv_template=conv_template,
                tokenizer=tokenizer,
                seq_length=seq_length,
                skip_first_response=skip_first_response,
            )
        self.dataset = []
        self.input_data = []
        for sample in dataset:
            item = self.format_tokenize_func(sample)
            if item is not None:
                idx = len(self.input_data)
                self.cached_data_dict[idx] = item
                self.dataset.append(item)
                self.input_data.append(sample)

        if debug and len(self.input_data) > 0:
            idx = 0
            logging_rank0(f"DEBUG data processing for sample index {idx}:\n", self.input_data[idx])
            logging_rank0(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        return self.dataset[i]
