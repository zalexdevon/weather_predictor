from classifier import logger
import pandas as pd
from classifier.entity.config_entity import DataTransformationConfig
from classifier.Mylib import myfuncs
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from classifier.Mylib import stringToObjectConverter


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Mã hóa các cột ordinal thành số

    Args:
        min_value (_type_): Giá trị nhỏ nhất khi mã hóa. Defaults to 0
    """

    def __init__(self, min_value=0) -> None:
        super().__init__()
        self.min_value = min_value

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        for col in X.columns:
            X[col] = X[col].cat.codes + self.min_value

        self.cols = X.columns.tolist()
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class DuringFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):

        # Lấy các cột numeric, nominal, ordinal
        (
            numeric_cols,
            numericcat_cols,
            cat_cols,
            binary_cols,
            nominal_cols,
            ordinal_cols,
        ) = myfuncs.get_different_types_feature_cols_from_df_14(X)

        numeric_cols = numeric_cols + numericcat_cols
        ordinal_cols = ordinal_cols + binary_cols

        nominal_cols_pipeline = Pipeline(
            steps=[
                ("1", OneHotEncoder(sparse_output=False, drop="first")),
                ("2", MinMaxScaler()),
            ]
        )

        ordinal_pipeline = Pipeline(
            steps=[
                ("1", CustomOrdinalEncoder()),
                ("2", MinMaxScaler()),
            ]
        )

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("1", MinMaxScaler(), numeric_cols),
                ("2", nominal_cols_pipeline, nominal_cols),
                ("3", ordinal_pipeline, ordinal_cols),
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

        return pd.DataFrame(
            X,
            columns=myfuncs.get_real_column_name_from_get_feature_names_out(
                self.column_transformer.get_feature_names_out()
            ),
        )

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_data(self):
        self.df_train = myfuncs.load_python_object(self.config.train_data_path)
        self.df_val = myfuncs.load_python_object(self.config.val_data_path)
        self.num_train_sample = len(self.df_train)

        self.feature_cols, self.target_col = (
            myfuncs.get_feature_cols_and_target_col_from_df_27(self.df_train)
        )

        # Load các transfomers
        self.list_before_feature_transformer = [
            stringToObjectConverter.convert_string_to_object_4(transformer)
            for transformer in self.config.list_before_feature_transformer
        ]

        self.list_after_feature_transformer = [
            stringToObjectConverter.convert_string_to_object_4(transformer)
            for transformer in self.config.list_after_feature_transformer
        ]

    def create_preprocessor_for_train_data(self):
        # TODO: d
        print("Start before_feature_pipeline")
        # d

        before_feature_pipeline = Pipeline(
            steps=[
                (str(index), transformer)
                for index, transformer in enumerate(
                    self.list_before_feature_transformer
                )
            ]
        )

        # TODO: d
        print("End before_feature_pipeline")
        # d

        after_feature_pipeline = Pipeline(
            steps=[
                (str(index), transformer)
                for index, transformer in enumerate(self.list_after_feature_transformer)
            ]
        )

        feature_pipeline = Pipeline(
            steps=[
                ("pre", before_feature_pipeline),
                ("during", DuringFeatureTransformer()),
                ("after", after_feature_pipeline),
            ]
        )

        target_pipeline = Pipeline(
            steps=[
                ("during", CustomOrdinalEncoder()),
            ]
        )

        column_transformer = ColumnTransformer(
            transformers=[
                ("feature", feature_pipeline, self.feature_cols),
                ("target", target_pipeline, [self.target_col]),
            ]
        )

        self.preprocessor = NamedColumnTransformer(column_transformer)

    def transform_data(self):
        df_train_transformed = self.preprocessor.fit_transform(self.df_train)
        df_val_transformed = self.preprocessor.transform(self.df_val)

        df_train_feature = df_train_transformed.drop(columns=[self.target_col])
        df_train_target = df_train_transformed[self.target_col]

        if self.config.do_smote == "t":
            smote = SMOTE(sampling_strategy="auto", random_state=42)
            df_train_feature, df_train_target = smote.fit_resample(
                df_train_feature, df_train_target
            )

        df_val_feature = df_val_transformed.drop(columns=[self.target_col])
        df_val_target = df_val_transformed[self.target_col]

        myfuncs.save_python_object(self.config.preprocessor_path, self.preprocessor)
        myfuncs.save_python_object(self.config.train_features_path, df_train_feature)
        myfuncs.save_python_object(self.config.train_target_path, df_train_target)
        myfuncs.save_python_object(self.config.val_features_path, df_val_feature)
        myfuncs.save_python_object(self.config.val_target_path, df_val_target)
