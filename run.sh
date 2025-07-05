export HTTP_PROXY=http://127.0.0.1:8890
export HTTPS_PROXY=http://127.0.0.1:8890
export WANDB_PROJECT=probing-housing
export WANDB_API_KEY=cf12a933e3e4edbc59fd0674ca4173fec75e11a4

python main.py \
    --model_name_or_path outputs/impute-pretrain/housing-roberta-tune-5e_4-mode3 \
    --config_name roberta-tune.json \
    --data_path datasets/housing/housing.csv \
    --column_info_path datasets/housing/column_info.json \
    --output_dir outputs/probing/housing-roberta-tune-5e_4-mode3 \
    --run_name housing-roberta-tune-5e_4-mode3 \
    --create_folds \
    --overwrite_output_dir \
    --report_to wandb \
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