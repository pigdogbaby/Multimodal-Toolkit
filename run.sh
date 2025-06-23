export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export WANDB_PROJECT=probing
export WANDB_API_KEY=cf12a933e3e4edbc59fd0674ca4173fec75e11a4

python main.py \
    --model_name_or_path outputs/tabred-pretrain/ecom-roberta-tune-random \
    --config_name roberta-tune.json \
    --data_path /home/wuyou/tabred/data/ecom-offers/ \
    --column_info_path datasets/income/column_info.json \
    --output_dir outputs/probing/roberta-tune-random-dbg \
    --run_name roberta-tune-5e_4-mode1 \
    --overwrite_output_dir \
    --report_to none \
    --task regression \
    --metric_for_best_model acc \
    --greater_is_better True \
    --load_best_model_at_end True \
    --categorical_encode_type label \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --weight_decay 1e-1 \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 30 \
    --save_total_limit 1 \
    --save_strategy steps \
    --save_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 100