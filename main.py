import logging
import os
import sys
from pprint import pformat
from statistics import mean, stdev
from typing import Callable, Dict

import numpy as np
import torch
from scipy.special import softmax
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    set_seed,
)

from evaluation import calc_classification_metrics, calc_regression_metrics, calc_imputation_metrics
from multimodal_transformers.data import load_data_from_folder, load_data_into_folds, TorchTabularTextDataset
from multimodal_transformers.model import AutoModelWithTabular, TabularConfig
from multimodal_transformers.multimodal_arguments import (
    ModelArguments,
    MultimodalDataTrainingArguments,
    OurTrainingArguments,
)
from util import create_dir_if_not_exists, get_args_info_as_str
from sklearn.preprocessing import StandardScaler
from torch import nn

os.environ["COMET_MODE"] = "DISABLED"
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, MultimodalDataTrainingArguments, OurTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    create_dir_if_not_exists(training_args.output_dir)
    stream_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(
        filename=os.path.join(training_args.output_dir, "train_log.txt"), mode="w+"
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[stream_handler, file_handler],
    )

    logger.info(f"======== Model Args ========\n{get_args_info_as_str(model_args)}\n")
    logger.info(f"======== Data Args ========\n{get_args_info_as_str(data_args)}\n")
    logger.info(
        f"======== Training Args ========\n{get_args_info_as_str(training_args)}\n"
    )

    tokenizer = None

    if not data_args.create_folds:
        train_dataset, val_dataset, test_dataset = load_data_from_folder(
            data_args.data_path,
            data_args.column_info["text_cols"],
            tokenizer,
            label_col=data_args.column_info["label_col"],
            label_list=data_args.column_info["label_list"],
            categorical_cols=data_args.column_info["cat_cols"],
            numerical_cols=data_args.column_info["num_cols"],
            categorical_encode_type=data_args.categorical_encode_type,
            categorical_handle_na=data_args.categorical_handle_na,
            categorical_na_value=data_args.categorical_na_value,
            numerical_transformer_method=data_args.numerical_transformer_method,
            numerical_handle_na=data_args.numerical_handle_na,
            numerical_how_handle_na=data_args.numerical_how_handle_na,
            numerical_na_value=data_args.numerical_na_value,
            sep_text_token_str=(
                tokenizer.sep_token
                if not data_args.column_info["text_col_sep_token"]
                else data_args.column_info["text_col_sep_token"]
            ),
            max_token_length=training_args.max_token_length,
            debug=training_args.debug_dataset,
            debug_dataset_size=training_args.debug_dataset_size,
            output_dir=training_args.output_dir,
        )
        train_datasets = [train_dataset]
        val_datasets = [val_dataset]
        test_datasets = [test_dataset]
    else:
        print("hello")
        train_datasets, val_datasets, test_datasets, cat_offsets = load_data_into_folds(
            data_args.data_path,
            data_args.num_folds,
            data_args.validation_ratio,
            data_args.column_info["text_cols"],
            tokenizer,
            label_col=data_args.column_info["label_col"],
            label_list=data_args.column_info["label_list"],
            categorical_cols=data_args.column_info["cat_cols"],
            numerical_cols=data_args.column_info["num_cols"],
            categorical_encode_type=data_args.categorical_encode_type,
            categorical_handle_na=data_args.categorical_handle_na,
            categorical_na_value=data_args.categorical_na_value,
            numerical_transformer_method=data_args.numerical_transformer_method,
            numerical_handle_na=data_args.numerical_handle_na,
            numerical_how_handle_na=data_args.numerical_how_handle_na,
            numerical_na_value=data_args.numerical_na_value,
            sep_text_token_str=None,
            max_token_length=training_args.max_token_length,
            debug=False,
            debug_dataset_size=training_args.debug_dataset_size,
            output_dir=training_args.output_dir,
        )
        print("check cat_offsets", cat_offsets)

    # X_num = np.load(data_args.data_path + "/X_num.npy")
    # num_means = np.nanmean(X_num, axis=0)
    # nan_indices = np.isnan(X_num)
    # X_num[nan_indices] = np.take(num_means, nan_indices.nonzero()[1])
    # scaler = StandardScaler()
    # X_num = scaler.fit_transform(X_num)

    # # X_bin = np.concatenate((np.load(data_args.data_path + "/X_bin.npy").astype(int), np.load(data_args.data_path + "/X_cat.npy").astype(int)), axis=1)
    # X_bin = np.load(data_args.data_path + "/X_bin.npy").astype(int)
    # # X_bin = np.load(data_args.data_path + "/X_cat.npy").astype(int)
    # num_bin = X_bin.shape[1]
    # cat_offsets = [X_bin[:,i].max() + 1 for i in range(num_bin)]
    # cat_offsets = [0] + cat_offsets
    # cat_offsets = np.cumsum(cat_offsets)
    # X_bin = X_bin + cat_offsets[:-1]

    # Y = np.load(data_args.data_path + "/Y.npy")
    # train_idx = np.load(data_args.data_path + "/split-default/train_idx.npy")
    # val_idx = np.load(data_args.data_path + "/split-default/val_idx.npy")
    # test_idx = np.load(data_args.data_path + "/split-default/test_idx.npy")

    # train_datasets = (TorchTabularTextDataset(
    #     encodings=None,
    #     categorical_feats=X_bin[train_idx],
    #     numerical_feats=X_num[train_idx],
    #     labels=Y[train_idx],
    #     df=None,
    #     label_list=None,
    # ),)
    # val_datasets = (TorchTabularTextDataset(
    #     encodings=None,
    #     categorical_feats=X_bin[val_idx],
    #     numerical_feats=X_num[val_idx],
    #     labels=Y[val_idx],
    #     df=None,
    #     label_list=None,
    # ),)
    # test_datasets = (TorchTabularTextDataset(
    #     encodings=None,
    #     categorical_feats=X_bin[test_idx],
    #     numerical_feats=X_num[test_idx],
    #     labels=Y[test_idx],
    #     df=None,
    #     label_list=None,
    # ),)

    train_dataset = train_datasets[0]

    set_seed(training_args.seed)
    task = data_args.task
    if task == "regression":
        num_labels = 1
    else:
        num_labels = (
            len(np.unique(train_dataset.labels))
            if data_args.num_classes == -1
            else data_args.num_classes
        )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # p.predictions is now a list of objects
            # The first entry is the actual predictions

            # I see, prediction = (logits, classifier_layer_outputs)!
            # print("dbg predictions", p.predictions[0].shape)
            # print("p.predictions[1].shape")
            # for i in p.predictions[1]:
            #     print(i.shape)

            # predictions = p.predictions[0]
            # if task_name == "classification":
            #     preds_labels = np.argmax(predictions, axis=1)
            #     if predictions.shape[-1] == 2:
            #         pred_scores = softmax(predictions, axis=1)[:, 1]
            #     else:
            #         pred_scores = softmax(predictions, axis=1)
            #     return calc_classification_metrics(
            #         pred_scores, preds_labels, p.label_ids
            #     )
            # elif task_name == "regression":
            #     preds = np.squeeze(predictions)
            #     return calc_regression_metrics(preds, p.label_ids)
            # else:
            #     return {}
            cat_logits, cat_labels, numerical_logits, numerical_labels = p.predictions[1]
            # print("metric", cat_labels[0])
            return calc_imputation_metrics(cat_logits, cat_labels, numerical_logits, numerical_labels)

        return compute_metrics_fn

    total_results = []
    for i, (train_dataset, val_dataset, test_dataset) in enumerate(
        zip(train_datasets, val_datasets, test_datasets)
    ):
        logger.info(f"======== Fold {i+1} ========")
        config = AutoConfig.from_pretrained(
            (
                model_args.config_name
                if model_args.config_name
                else model_args.model_name_or_path
            ),
            cache_dir=model_args.cache_dir,
        )
        print("cat_feats", train_dataset.cat_feats.shape)
        print("numerical_feats", train_dataset.numerical_feats.shape)
        print("labels", train_dataset.labels.shape)
        tabular_config = TabularConfig(
            num_labels=num_labels,
            cat_feat_dim=(
                train_dataset.cat_feats.shape[1]
                if train_dataset.cat_feats is not None
                else 0
            ),
            numerical_feat_dim=(
                train_dataset.numerical_feats.shape[1]
                if train_dataset.numerical_feats is not None
                else 0
            ),
            cat_offsets=cat_offsets,
            num_feats=train_dataset.cat_feats.shape[1]+train_dataset.numerical_feats.shape[1],
            **vars(data_args),
        )
        config.tabular_config = tabular_config

        model = AutoModelWithTabular.from_config(
            config=config
        )
        if i == 0:
            logger.info(tabular_config)
            logger.info(model)
            for name, module in model.named_modules():
                if isinstance(module, nn.Module):
                    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    print(f"Module: {name}, Parameters: {num_params}")
            num_params = sum(p.numel() for p in model.parameters())
            print("sum_params", num_params)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=build_compute_metrics_fn(task),
        )
        if training_args.do_train:
            trainer.train(
                resume_from_checkpoint=(
                    model_args.model_name_or_path
                    if os.path.isdir(model_args.model_name_or_path)
                    else None
                )
            )
            trainer.save_model()

        # Evaluation
        eval_results = {}
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            eval_result = trainer.evaluate(eval_dataset=val_dataset)
            logger.info(pformat(eval_result, indent=4))

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_metric_results_{task}_fold_{i+1}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(task))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

        # if training_args.do_predict:
        #     logging.info("*** Test ***")

        #     predictions = trainer.predict(test_dataset=test_dataset).predictions[0]
        #     output_test_file = os.path.join(
        #         training_args.output_dir, f"test_results_{task}_fold_{i+1}.txt"
        #     )
        #     eval_result = trainer.evaluate(eval_dataset=test_dataset)
        #     logger.info(pformat(eval_result, indent=4))
        #     if trainer.is_world_process_zero():
        #         with open(output_test_file, "w") as writer:
        #             logger.info("***** Test results {} *****".format(task))
        #             writer.write("index\tprediction\n")
        #             if task == "classification":
        #                 predictions = np.argmax(predictions, axis=1)
        #             for index, item in enumerate(predictions):
        #                 if task == "regression":
        #                     writer.write(
        #                         "%d\t%3.3f\t%d\n"
        #                         % (index, item, test_dataset.labels[index])
        #                     )
        #                 else:
        #                     item = test_dataset.get_labels()[item]
        #                     writer.write("%d\t%s\n" % (index, item))
        #         output_test_file = os.path.join(
        #             training_args.output_dir,
        #             f"test_metric_results_{task}_fold_{i+1}.txt",
        #         )
        #         with open(output_test_file, "w") as writer:
        #             logger.info("***** Test results {} *****".format(task))
        #             for key, value in eval_result.items():
        #                 logger.info("  %s = %s", key, value)
        #                 writer.write("%s = %s\n" % (key, value))
        #         eval_results.update(eval_result)
        del model
        del config
        del tabular_config
        del trainer
        torch.cuda.empty_cache()
        total_results.append(eval_results)
    aggr_res = aggregate_results(total_results)
    logger.info("========= Aggr Results ========")
    logger.info(pformat(aggr_res, indent=4))

    output_aggre_test_file = os.path.join(
        training_args.output_dir, f"all_test_metric_results_{task}.txt"
    )
    with open(output_aggre_test_file, "w") as writer:
        logger.info("***** Aggr results {} *****".format(task))
        for key, value in aggr_res.items():
            logger.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))


def aggregate_results(total_test_results):
    metric_keys = list(total_test_results[0].keys())
    aggr_results = dict()

    for metric_name in metric_keys:
        if type(total_test_results[0][metric_name]) is str:
            continue
        res_list = []
        for results in total_test_results:
            res_list.append(results[metric_name])
        if len(res_list) == 1:
            metric_avg = res_list[0]
            metric_stdev = 0
        else:
            metric_avg = mean(res_list)
            metric_stdev = stdev(res_list)

        aggr_results[metric_name + "_mean"] = metric_avg
        aggr_results[metric_name + "_stdev"] = metric_stdev
    return aggr_results


if __name__ == "__main__":
    main()
