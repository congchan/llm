cd "$(dirname "${BASH_SOURCE[0]}")/../llm/alignment/summarize_rlhf/"


num_processes=7
config_file=configs/accelerate/default_config_zero2.yaml
export LAUNCHER="accelerate launch \
    --config_file $config_file \
    --num_processes $num_processes \
    "

output_dir=outputs/tldr
trl_config_file=configs/ppo_tldr.yml
n_eval_prompts=2048
rw_model_batch_size=28
rw_model_name_or_path=reward_model/outputs/summarize_tldr_rm/checkpoint-best/
dataset_name=data/prompts/

export CMD=" \
    $LAUNCHER trlx_tlr.py \
    --output_dir $output_dir \
    --trl_config_file $trl_config_file \
    --n_eval_prompts $n_eval_prompts \
    --dataset_name $dataset_name \
    --rw_model_batch_size $rw_model_batch_size \
    --rw_model_name_or_path $rw_model_name_or_path \
    "

echo $CMD

$CMD