#/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/../llm/alignment/summarize_rlhf/reward_model/"

deepspeed_config_file=ds_config_zero2.json
dataset_name="openai/summarize_from_feedback"
per_device_train_batch_size=1
gradient_accumulation_steps=1
sft_model_name_or_path="CarperAI/openai_summarize_tldr_sft"
how_layers_unfrozen=0.6
output_dir=outputs/summarize_tldr_rm
num_train_epochs=2
do_shuffle=0

export LAUNCHER="deepspeed "
export CMD=" \
    $LAUNCHER train_reward_model.py  \
    --deepspeed_config_file $deepspeed_config_file \
    --dataset_name $dataset_name \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --sft_model_name_or_path $sft_model_name_or_path \
    --how_layers_unfrozen $how_layers_unfrozen \
    --num_train_epochs $num_train_epochs \
    --do_shuffle $do_shuffle \
    --output_dir $output_dir \
    "

echo $CMD

$CMD