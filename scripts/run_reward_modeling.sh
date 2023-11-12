#/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/llm/alignment/reward_modeling/"

deepspeed_config_file=ds_config_zero2.json
dataset_name=data/reward_dataset/
per_device_train_batch_size=1
gradient_accumulation_steps=1
sft_model_name_or_path=IDEA-CCNL/Ziya-LLaMA-7B-Reward
how_layers_unfrozen=0.6
output_dir=outputs
num_train_epochs=2
do_shuffle=0

export LAUNCHER="deepspeed  "
export CMD=" \
    $LAUNCHER train_reward.py  \
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