# LLM(Large Language Modeling)
A platform for training large language model based multi-turns chatbots.

Developed based on FastChat, trl, trlx, Huggingface transformers.

## Features
* Support training llm models.
* Characters awared templates.
* Long sequence length training with [FlashAttention2](https://github.com/HazyResearch/flash-attention). 
* A stacked dataset class to support long chats training. 
* Ghost Attention (GAtt) in templates.

## Installation
```python
pip3 install -e ".[data]"
```

This project make use of transformers trainer to manage training process, and deepspeed to handle data parallelism and zero(1, 2, 3) optimization. Please install transformers and `deepspeed>=0.10.3`.

The project has fully tested with `deepspeed>=0.10.3`, older version of deepspeed may persist unexpected bugs and error.

Since [transformers>=4.34.0](https://github.com/huggingface/transformers/releases), huggingface transformers support flash-attention2 integrated in Llama and Falcon series models. I recommend using `transformers>=4.34.0` instead of wrting extra patches.

Please go to [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) and install flash-attention2, or just run below command to install it.
```
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE  pip3 install flash_attn  --no-build-isolation
```

Some models(such as [Baichuan2](https://github.com/baichuan-inc/Baichuan2)) support [xformers](https://github.com/facebookresearch/xformers), you need to install it by yourself.
```
pip3 install -U xformers
```


# Pre-training

# Fine-tuning
1. Process data
    ```python
    python -m llm.data.process_chats --dataset PIPPA
    ```

2. The example training script is `run_finetune.sh`, please replace args values with your actual need. Depend on different hardware and clusters, you will need to handle multi-nodes working by yourself.


# Alignment
I have reproduce Stiennon, Nisan, et al. Learning to Summarize from Human Feedback. arXiv:2009.01325, arXiv, 15 Feb. 2022. arXiv.org, http://arxiv.org/abs/2009.01325.

## Learning to Summarize from Human Feedback
1. Train SFT: I skipped sft and make use of open-sourced as sft's ability is well tested and demonstrated.
    
    Checkpoint: [SFT](https://huggingface.co/CarperAI/openai_summarize_tldr_sft)


2. Train Reward Model:
    ```
    cd scripts
    bash run_summarize_reward_modeling.sh
    ```

3. PPO training: note that the code only support single node and multi-gpus training. I use 7 gpus for PPO training and 1 gpu for reward model serving.
    ```
    cd scripts
    bash run_summarize_rlhf.sh
    ```


## General assistant alignment



# Citation
* Stiennon, Nisan, et al. Learning to Summarize from Human Feedback. arXiv:2009.01325, arXiv, 15 Feb. 2022. arXiv.org, http://arxiv.org/abs/2009.01325.
* https://github.com/lm-sys/FastChat
* https://github.com/CarperAI/trlx
* https://github.com/lvwerra/trl/
* https://github.com/huggingface/transformers