export WANDB_PROJECT=income-imputation-minmax
export WANDB_API_KEY=cf12a933e3e4edbc59fd0674ca4173fec75e11a4

python main.py \
    --model_name_or_path None \
    --config_name pt-tiny.json \
    --data_path datasets/income/income.csv \
    --column_info_path datasets/income/column_info.json \
    --output_dir outputs/income-imputation-minmax/pt-tiny-hard-30ep-3e_3 \
    --run_name pt-tiny-hard-30ep-3e_3 \
    --overwrite_output_dir \
    --report_to wandb \
    --task classification \
    --create_folds \
    --categorical_encode_type label \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.015 \
    --learning_rate 3e-3 \
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