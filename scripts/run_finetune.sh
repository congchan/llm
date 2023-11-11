PYTHON="torchrun"
export LAUNCHER="
    $PYTHON \
    "


model_name_or_path=Yi-34B
data_path=train.json
eval_data_path=valid.json
output_dir=run
conv_template=mtagent
num_train_epochs=4
per_device_train_batch_size=1
per_device_eval_batch_size=4
gradient_accumulation_steps=4
save_strategy=epoch
save_total_limit=4
evaluation_strategy=steps
eval_steps=50
learning_rate=2e-6
weight_decay=0.
warmup_ratio=0.01
lr_scheduler_type=cosine
logging_steps=1
model_max_length=8192

config_json=train/ds_config_zero3.json
DEEPSPEED_ARGS=" \
    --deepspeed ${config_json} \
    "


export CMD=" \
    $LAUNCHER train/train_yi.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path  \
    --eval_data_path $eval_data_path \
    --conv_template $conv_template \
    --output_dir $output_dir \
    --num_train_epochs $num_train_epochs    \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size  \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --save_strategy $save_strategy \
    --save_total_limit $save_total_limit \
    --evaluation_strategy $evaluation_strategy \
    --eval_steps $eval_steps \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay  \
    --warmup_ratio $warmup_ratio  \
    --lr_scheduler_type $lr_scheduler_type \
    --logging_steps $logging_steps  \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --model_max_length $model_max_length \
     $DEEPSPEED_ARGS \
    "

echo $CMD

$CMD