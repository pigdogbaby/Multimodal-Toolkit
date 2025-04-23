export WANDB_PROJECT=sweeps_demo
export WANDB_API_KEY=###

python main.py \
    --model_name_or_path None \
    --config_name roberta-tune.json \
    --data_path /home/wuyou/tabred/data/sberbank-housing/ \
    --column_info_path datasets/income/column_info.json \
    --output_dir outputs/sweep/ \
    --run_name roberta-tune-5e_4-test \
    --overwrite_output_dir \
    --report_to wandb \
    --task regression \
    --metric_for_best_model rmse \
    --greater_is_better False \
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
    --num_train_epochs 1000 \
    --save_total_limit 1 \
    --save_strategy epoch \
    --evaluation_strategy epoch \