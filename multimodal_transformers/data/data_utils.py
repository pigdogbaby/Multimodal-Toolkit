import logging
import types

import numpy as np
import pandas as pd
from typing import List, Callable
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, MinMaxScaler

logger = logging.getLogger(__name__)


class CategoricalFeatures:
    """Class to help encode categorical features
    From https://github.com/abhishekkrthakur/mlframework/blob/master/src/categorical.py
    """

    def __init__(
        self,
        categorical_cols: List[str],
        encoding_type: str,
        handle_na: bool = False,
        na_value: str = "-9999999",
        ohe_handle_unknown: str = "error",
    ):
        """
        Args:
            categorical_cols (List[str]):
                the column names in the dataset that contain categorical
                features
            encoding_type (str): method we want to preprocess our categorical
            features.
                choices: [ 'ohe', 'binary', None]
            handle_na (bool): whether to handle nan by treating them as a
                separate categorical value
            na_value (string): what the nan values should be converted to
            ohe_handle_unknown (string): How the one hot encoder should handle
                unknown values it encounters.
                Ref: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        """
        self.cat_cols = categorical_cols
        self.enc_type = encoding_type
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None
        self.handle_na = handle_na
        self.na_value = '-99999'
        self.feat_names = []
        self.ohe_handle_unknown = ohe_handle_unknown

    def _label_encoding(self, dataframe: pd.DataFrame):
        cat_offsets = []
        for c in self.cat_cols:
            lbl = preprocessing.LabelEncoder()
            # additionally encode na_value
            tmp = dataframe[c].astype(str)
            tmp.loc[len(tmp)] = self.na_value
            lbl.fit((tmp.values))
            self.label_encoders[c] = lbl
            print("label_encoding_classes", np.unique(tmp.values), lbl.transform(np.unique(tmp.values)))
            cat_offsets.append(len(lbl.classes_))
        # print("dbg cat_offsets", cat_offsets)
        return cat_offsets

    def _label_binarization(self, dataframe: pd.DataFrame):
        for c in self.cat_cols:
            dataframe[c] = dataframe[c].astype(str)
            lb = preprocessing.LabelBinarizer()
            lb.fit(dataframe[c].values)
            self.binary_encoders[c] = lb

            # Create new class names
            for class_name in lb.classes_:
                new_col_name = f"{c}__{change_name_func(class_name)}"
                self.feat_names.append(new_col_name)
                if len(lb.classes_) == 2:
                    break

    def _one_hot(self, dataframe: pd.DataFrame):
        self.ohe = preprocessing.OneHotEncoder(
            sparse_output=False, handle_unknown=self.ohe_handle_unknown
        )
        self.ohe.fit(dataframe[self.cat_cols].values)
        self.feat_names = list(self.ohe.get_feature_names_out(self.cat_cols))

    def nan_handler(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for c in self.cat_cols:
            dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna(self.na_value)
        return dataframe

    def fit(self, dataframe: pd.DataFrame):
        if self.handle_na:
            dataframe = self.nan_handler(dataframe)
        # print("check2\n", dataframe.head())
        # print(dataframe.columns)
        if self.enc_type == "label":
            return self._label_encoding(dataframe)
        elif self.enc_type == "binary":
            self._label_binarization(dataframe)
        elif self.enc_type == "ohe":
            self._one_hot(dataframe)
        elif self.enc_type is None or self.enc_type == "none":
            logger.info(f"Encoding type is none, no action taken.")
        else:
            raise Exception("Encoding type not understood")

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.fit(dataframe)
        return self.transform(dataframe)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.astype(str)
        if self.handle_na:
            dataframe = self.nan_handler(dataframe)

        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe

        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)
                class_names = [
                    f"{c}__{lbl.classes_[j]}_binary" for j in range(val.shape[1])
                ]
                val = pd.DataFrame(val, columns=class_names, index=dataframe.index)
                dataframe = pd.concat([dataframe, val], axis=1)
            return dataframe

        elif self.enc_type == "ohe":
            val = self.ohe.transform(dataframe[self.cat_cols].values)
            new_df = {}
            for j in range(val.shape[1]):
                new_df[self.feat_names[j]] = val[:, j]
            return pd.DataFrame(new_df, index=dataframe.index)

        elif self.enc_type is None or self.enc_type == "none":
            logger.info(f"Encoding type is none, no action taken.")
            return dataframe[self.cat_cols].values
        else:
            raise Exception("Encoding type not understood")


class NumericalFeatures:
    """Class to help encode numerical features"""

    def __init__(
        self,
        numerical_cols: List[str],
        numerical_transformer_method: str,
        handle_na: bool = False,
        how_handle_na: str = "value",
        na_value: float = 0.0,
    ):
        """

        Args:
            numerical_cols (List[str]): numerical column names in the dataset
            numerical_transformer_method (str): what transformation to use
            handle_na (bool, optional): whether to handle NA. Defaults to False.
            how_handle_na (str, optional): How to handle NA. Defaults to
                "value" which replaces with na_value. You can also use "mean"
                or "median".
            na_value (float, optional): Value to replace NA with if
                how_handle_na="value". Defaults to 0.0.
        """
        self.num_cols = numerical_cols
        self.numerical_transformer_method = numerical_transformer_method
        self.numerical_transformer = None
        self.handle_na = handle_na
        self.how_handle_na = how_handle_na
        self.na_value = na_value
        self.scaler = MinMaxScaler()
        self.means = None

    def nan_handler(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        print("dbg numerical nan_handler", np.isnan(dataframe[self.num_cols].values).sum(), dataframe[self.num_cols].values.size)
        if self.means is None:
            means = dataframe[self.num_cols].mean()
            self.means = means
        else:
            means = self.means
        print(means, means.tolist())        
        dataframe.loc[:, self.num_cols] = dataframe[self.num_cols].astype(float).fillna(
            dict(means), inplace=False
        )
        # if self.how_handle_na == "median":
        #     dataframe.loc[:, self.num_cols] = dataframe[self.num_cols].fillna(
        #         dict(dataframe[self.num_cols].median()), inplace=False
        #     )
        # elif self.how_handle_na == "mean":
        #     dataframe.loc[:, self.num_cols] = dataframe[self.num_cols].fillna(
        #         dict(dataframe[self.num_cols].mean()), inplace=False
        #     )
        # elif self.how_handle_na == "value":
        #     dataframe.loc[:, self.num_cols] = dataframe[self.num_cols].fillna(
        #         self.na_value, inplace=False
        #     )
        # else:
        #     raise ValueError(f"Unknown NaN handling method {self.how_handle_na}.")
        return dataframe

    def fit(self, dataframe: pd.DataFrame):
        dataframe = self.nan_handler(dataframe[self.num_cols])
        print("fit")
        self.scaler.fit(dataframe)
        # Build transformer
        # if self.numerical_transformer_method == "yeo_johnson":
        #     self.numerical_transformer = PowerTransformer(method="yeo-johnson")
        # elif self.numerical_transformer_method == "box_cox":
        #     self.numerical_transformer = PowerTransformer(method="box-cox")
        # elif self.numerical_transformer_method == "quantile_normal":
        #     self.numerical_transformer = QuantileTransformer(
        #         output_distribution="normal"
        #     )
        # else:
        #     raise ValueError(
        #         f"preprocessing transformer method "
        #         f"{self.numerical_transformer_method} not implemented"
        #     )
        # Fit transformer
        # num_feats = dataframe[self.num_cols]
        # missing_values_count = (num_feats == '?').sum().sum()
        # print(missing_values_count)
        # self.numerical_transformer.fit(num_feats)

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.fit(dataframe)
        return self.transform(dataframe)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        print("detect nan", np.any(np.isnan(dataframe[self.num_cols])))
        dataframe = self.nan_handler(dataframe[self.num_cols])
        print("transform", dataframe.head())
        stand_data = self.scaler.transform(dataframe)
        stand_dataframe = pd.DataFrame(stand_data, columns=dataframe.columns, index=dataframe.index)
        print("check StandardScaler", stand_dataframe.head())
        # return self.numerical_transformer.transform(dataframe[self.num_cols])
        return stand_dataframe


def change_name_func(x: str) -> str:
    return x.lower().replace(", ", "_").replace(" ", "_")


def convert_to_func(container_arg):
    """convert container_arg to function that returns True if an element is in container_arg"""
    if container_arg is None:
        return lambda df, x: False
    if not isinstance(container_arg, types.FunctionType):
        assert type(container_arg) is list or type(container_arg) is set
        return lambda df, x: x in container_arg
    else:
        return container_arg


def agg_text_columns_func(
    empty_row_values: List[str], replace_text: str, texts: List[str]
) -> List[str]:
    """replace empty texts or remove empty text str from a list of text str"""
    processed_texts = []
    for text in texts.astype("str"):
        if text not in empty_row_values:
            processed_texts.append(text)
        else:
            if replace_text is not None:
                processed_texts.append(replace_text)
    return processed_texts


def get_matching_cols(
    df: pd.DataFrame, col_match_func: Callable[[pd.DataFrame, str], bool]
) -> List[str]:
    return [c for c in df.columns if col_match_func(df, c)]
