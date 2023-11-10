# LLM(Large Language Modeling)
A platform for training large language model based multi-turns chatbots.

Developed based on FastChat, trl, trlx, Huggingface transformers.

## Features
* Support Llama, Baichuan, Qwen seriels models.
* Characters awared templates.
* Long sequence length training with [FlashAttention2](https://github.com/HazyResearch/flash-attention). 
* A stacked dataset class to support long chats training. 
* Ghost Attention (GAtt) in templates.

## Installation
```python
pip3 install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --upgrade -e ".[data]"
```

# Pre-training

# Fine-tuning
1. Process data
    ```python
    python -m llm.data.process_chats --dataset PIPPA
    ```


# Alignment
