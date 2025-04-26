from classifier import logger
import pandas as pd
from classifier.entity.config_entity import DataTransformationConfig
from Mylib import myfuncs
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from Mylib import stringToObjectConverter


class DuringFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_ordinal_dict) -> None:
        super().__init__()
        self.feature_ordinal_dict = feature_ordinal_dict

    def fit(self, X, y=None):
        # Lấy các cột numeric, nominal, ordinal
        (
            numeric_cols,
            numericcat_cols,
            _,
            _,
            nominal_cols,
            _,
        ) = myfuncs.get_different_types_feature_cols_from_df_14(X)

        numeric_cols = numeric_cols + numericcat_cols

        ordinal_binary_cols = list(self.feature_ordinal_dict.keys())

        nominal_cols_pipeline = Pipeline(
            steps=[
                ("1", OneHotEncoder(sparse_output=False, drop="first")),
                ("2", MinMaxScaler()),
            ]
        )

        ordinal_binary_cols_pipeline = Pipeline(
            steps=[
                (
                    "1",
                    OrdinalEncoder(categories=list(self.feature_ordinal_dict.values())),
                ),
                ("2", MinMaxScaler()),
            ]
        )

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("1", MinMaxScaler(), numeric_cols),
                ("2", nominal_cols_pipeline, nominal_cols),
                ("3", ordinal_binary_cols_pipeline, ordinal_binary_cols),
            ],
        )

        self.column_transformer.fit(X)

    def transform(self, X, y=None):
        X = self.column_transformer.transform(X)

        self.cols = myfuncs.get_real_column_name_from_get_feature_names_out(
            self.column_transformer.get_feature_names_out()
        )

        return pd.DataFrame(X, columns=self.cols)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class NamedColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_transformer) -> None:
        super().__init__()
        self.column_transformer = column_transformer

    def fit(self, X, y=None):
        self.column_transformer.fit(X)

    def transform(self, X, y=None):
        X = self.column_transformer.transform(X)

        cols = myfuncs.fix_name_by_LGBM_standard(
            myfuncs.get_real_column_name_from_get_feature_names_out(
                self.column_transformer.get_feature_names_out()
            )
        )

        return pd.DataFrame(
            X,
            columns=cols,
        )

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_data(self):
        # TODO: d
        print(f"data correction path: {self.config.train_data_path}")

        # d

        self.df_train = myfuncs.load_python_object(self.config.train_data_path)
        self.feature_ordinal_dict = myfuncs.load_python_object(
            self.config.feature_ordinal_dict_path
        )

        # TODO: d
        print(f"feature_ordinal_dict: {self.feature_ordinal_dict}")
        # d

        self.correction_transformer = myfuncs.load_python_object(
            self.config.correction_transformer_path
        )

        # TODO: d
        print(f"correction_transformer: {self.config.correction_transformer_path}")
        # d

        self.df_val = myfuncs.load_python_object(self.config.val_data_path)

        self.num_train_sample = len(self.df_train)

        self.feature_cols, self.target_col = (
            myfuncs.get_feature_cols_and_target_col_from_df_27(self.df_train)
        )

        # Load các transfomers
        self.list_after_feature_transformer = [
            stringToObjectConverter.convert_complex_MLmodel_yaml_to_object(transformer)
            for transformer in self.config.list_after_feature_transformer
        ]

    def create_preprocessor_for_train_data(self):
        after_feature_pipeline = (
            Pipeline(
                steps=[
                    (str(index), transformer)
                    for index, transformer in enumerate(
                        self.list_after_feature_transformer
                    )
                ]
            )
            if len(self.list_after_feature_transformer) > 0
            else Pipeline(steps=[("passthrough", "passthrough")])
        )

        feature_pipeline = Pipeline(
            steps=[
                ("during", DuringFeatureTransformer(self.feature_ordinal_dict)),
                ("after", after_feature_pipeline),
            ]
        )

        column_transformer = ColumnTransformer(
            transformers=[
                ("feature", feature_pipeline, self.feature_cols),
                ("target", OrdinalEncoder(), [self.target_col]),
            ]
        )

        self.transformation_transformer = NamedColumnTransformer(column_transformer)

    def transform_data(self):
        df_train_transformed = self.transformation_transformer.fit_transform(
            self.df_train
        )
        df_train_feature = df_train_transformed.drop(columns=[self.target_col]).astype(
            "float32"
        )
        df_train_target = df_train_transformed[self.target_col].astype("int8")

        if self.config.do_smote == "t":
            smote = SMOTE(sampling_strategy="auto", random_state=42)
            df_train_feature, df_train_target = smote.fit_resample(
                df_train_feature, df_train_target
            )

        df_val_corrected = self.correction_transformer.transform(
            self.df_val, data_type="test"
        )
        df_val_transformed = self.transformation_transformer.transform(df_val_corrected)
        df_val_feature = df_val_transformed.drop(columns=[self.target_col]).astype(
            "float32"
        )
        df_val_target = df_val_transformed[self.target_col].astype("int8")

        class_names = list(
            self.transformation_transformer.column_transformer.named_transformers_[
                "target"
            ].categories_
        )

        # TODO: d
        print(f"class_names: {class_names}")
        # d

        myfuncs.save_python_object(
            self.config.transformation_transformer_path, self.transformation_transformer
        )
        myfuncs.save_python_object(self.config.train_features_path, df_train_feature)
        myfuncs.save_python_object(self.config.train_target_path, df_train_target)
        myfuncs.save_python_object(self.config.val_features_path, df_val_feature)
        myfuncs.save_python_object(self.config.val_target_path, df_val_target)
        myfuncs.save_python_object(self.config.val_target_path, df_val_target)
        myfuncs.save_python_object(self.config.class_names_path, class_names)
