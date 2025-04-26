from classifier import logger
import pandas as pd
from classifier.entity.config_entity import DataCorrectionConfig
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
import re
from sklearn.impute import SimpleImputer


# Chuỗi nào có PM thì giá trị giờ cộng thêm 12 phút
def process_time(gold_time: str):
    if gold_time.endswith("PM"):
        parts = gold_time.split(":")
        hour = int(parts[0]) + 12
        gold_time = f"{hour}:{parts[1]}"

    res = re.split("(AM|PM)", gold_time)[0].strip()
    res = res.split(":")[0]
    return res


class BeforeHandleMissingValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        # Xóa các cột không cần thiết
        df = df.drop(
            columns=[
                "country",
                "location_name",
                "latitude",
                "longitude",
                "timezone",
                "last_updated_epoch",
                "last_updated",
                "temperature_celsius",
                "temperature_fahrenheit",
                "condition_text",
                "wind_mph",
                "wind_direction",
                "pressure_mb",
                "precip_mm",
                "precip_in",
                "feels_like_celsius",
                "feels_like_fahrenheit",
                "visibility_miles",
                "gust_mph",
                "air_quality_us-epa-index",
                "air_quality_gb-defra-index",
                "sunrise",
                "sunset",
                "moonrise",
                "moonset",
            ]
        )

        #  Đổi tên cột
        rename_dict = {
            "wind_kph": "wind_kph_num",
            "wind_degree": "wind_degree_num",
            "pressure_in": "pressure_in_num",
            "humidity": "humidity_num",
            "cloud": "cloud_num",
            "visibility_km": "visibility_km_num",
            "uv_index": "uv_index_num",
            "gust_kph": "gust_kph_num",
            "air_quality_Carbon_Monoxide": "air_quality_Carbon_Monoxide_num",
            "air_quality_Ozone": "air_quality_Ozone_num",
            "air_quality_Nitrogen_dioxide": "air_quality_Nitrogen_dioxide_num",
            "air_quality_Sulphur_dioxide": "air_quality_Sulphur_dioxide_num",
            "air_quality_PM2.5": "air_quality_PM2_5_num",
            "air_quality_PM10": "air_quality_PM10_num",
            "moon_phase": "moon_phase_nom",
            "moon_illumination": "moon_illumination_num",
            "temp_bin": "temp_bin_target",
        }

        df = df.rename(columns=rename_dict)

        # Sắp xếp các cột theo đúng thứ tự
        (
            numeric_cols,
            numericCat_cols,
            cat_cols,
            binary_cols,
            nominal_cols,
            ordinal_cols,
            target_col,
        ) = myfuncs.get_different_types_cols_from_df_4(df)

        df = df[
            numeric_cols
            + numericCat_cols
            + binary_cols
            + nominal_cols
            + ordinal_cols
            + [target_col]
        ]

        self.cols = df.columns

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class HandleMissingValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        df = X

        (
            self.numeric_cols,
            self.numericCat_cols,
            self.cat_cols,
            _,
            _,
            _,
            self.target_col,
        ) = myfuncs.get_different_types_cols_from_df_4(df)

        self.missing_value_transformer = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="mean"), self.numeric_cols),
                (
                    "numCat",
                    SimpleImputer(strategy="most_frequent"),
                    self.numericCat_cols,
                ),
                ("cat", SimpleImputer(strategy="most_frequent"), self.cat_cols),
                ("target", SimpleImputer(strategy="most_frequent"), [self.target_col]),
            ]
        )
        self.missing_value_transformer.fit(df)

    def transform(self, X, y=None):
        df = X
        df = self.missing_value_transformer.transform(df)

        self.cols = (
            self.numeric_cols + self.numericCat_cols + self.cat_cols + [self.target_col]
        )

        return pd.DataFrame(
            X,
            columns=self.cols,
        )

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class AfterHandleMissingValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        numeric_cols, numericCat_cols, cat_cols, _, _, _, target_col = (
            myfuncs.get_different_types_cols_from_df_4(df)
        )

        # Chuyển đổi về đúng kiểu dữ liệu
        df[numeric_cols] = df[numeric_cols].astype("float32")
        df[numericCat_cols] = df[numericCat_cols].astype("float32")
        df[cat_cols] = df[cat_cols].astype("category")
        df[target_col] = df[target_col].astype("category")

        # Loại bỏ duplicates
        df = df.drop_duplicates().reset_index()

        self.cols = df.columns

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class DataCorrectorForTrainAndTest:
    def __init__(self):
        pass

    def transform(self, df, data_type):
        # TODO: d
        print("Chả nhẽ là lại chạy ở đây được sao")
        # d

        # Xóa các cột không cần thiết
        df = df.drop(
            columns=[
                "country",
                "location_name",
                "latitude",
                "longitude",
                "timezone",
                "last_updated_epoch",
                "last_updated",
                "temperature_celsius",
                "temperature_fahrenheit",
                "condition_text",
                "wind_mph",
                "wind_direction",
                "pressure_mb",
                "precip_mm",
                "precip_in",
                "feels_like_celsius",
                "feels_like_fahrenheit",
                "visibility_miles",
                "gust_mph",
                "air_quality_us-epa-index",
                "air_quality_gb-defra-index",
                "sunrise",
                "sunset",
                "moonrise",
                "moonset",
            ]
        )

        #  Đổi tên cột
        rename_dict = {
            "wind_kph": "wind_kph_num",
            "wind_degree": "wind_degree_num",
            "pressure_in": "pressure_in_num",
            "humidity": "humidity_num",
            "cloud": "cloud_num",
            "visibility_km": "visibility_km_num",
            "uv_index": "uv_index_num",
            "gust_kph": "gust_kph_num",
            "air_quality_Carbon_Monoxide": "air_quality_Carbon_Monoxide_num",
            "air_quality_Ozone": "air_quality_Ozone_num",
            "air_quality_Nitrogen_dioxide": "air_quality_Nitrogen_dioxide_num",
            "air_quality_Sulphur_dioxide": "air_quality_Sulphur_dioxide_num",
            "air_quality_PM2.5": "air_quality_PM2_5_num",
            "air_quality_PM10": "air_quality_PM10_num",
            "moon_phase": "moon_phase_nom",
            "moon_illumination": "moon_illumination_num",
            "temp_bin": "temp_bin_target",
        }

        df = df.rename(columns=rename_dict)

        # Sắp xếp các cột theo đúng thứ tự
        (
            numeric_cols,
            numericCat_cols,
            cat_cols,
            binary_cols,
            nominal_cols,
            ordinal_cols,
            target_col,
        ) = myfuncs.get_different_types_cols_from_df_4(df)

        df = df[
            numeric_cols
            + numericCat_cols
            + binary_cols
            + nominal_cols
            + ordinal_cols
            + [target_col]
        ]

        # Xử lí missing value
        if data_type == "train":
            self.missing_value_transformer = ColumnTransformer(
                transformers=[
                    ("num", SimpleImputer(strategy="mean"), numeric_cols),
                    (
                        "numCat",
                        SimpleImputer(strategy="most_frequent"),
                        numericCat_cols,
                    ),
                    ("cat", SimpleImputer(strategy="most_frequent"), cat_cols),
                    ("target", SimpleImputer(strategy="most_frequent"), [target_col]),
                ]
            )
            self.missing_value_transformer.fit(df)

        df = self.missing_value_transformer.transform(df)
        df = pd.DataFrame(
            df, columns=numeric_cols + numericCat_cols + cat_cols + [target_col]
        )

        # Chuyển đổi về đúng kiểu dữ liệu
        df[numeric_cols] = df[numeric_cols].astype("float32")
        df[numericCat_cols] = df[numericCat_cols].astype("float32")
        df[cat_cols] = df[cat_cols].astype("category")
        df[target_col] = df[target_col].astype("category")

        # Loại bỏ duplicates
        df = df.drop_duplicates().reset_index()

        return df


FEATURE_ORDINAL_DICT = {}


class DataCorrection:
    def __init__(self, config: DataCorrectionConfig):
        self.config = config

    def load_data(self):
        self.df = myfuncs.load_python_object(self.config.train_data_path)

    def create_preprocessor_for_train_data(self):
        self.transformer = Pipeline(
            steps=[
                ("1", BeforeHandleMissingValueTransformer()),
                ("2", HandleMissingValueTransformer()),
                ("3", AfterHandleMissingValueTransformer()),
            ]
        )

    def transform_data(self):
        df = self.transformer.fit_transform(self.df)

        myfuncs.save_python_object(self.config.data_path, df)
        myfuncs.save_python_object(
            self.config.feature_ordinal_dict_path, FEATURE_ORDINAL_DICT
        )
        myfuncs.save_python_object(
            self.config.correction_transformer_path, self.transformer
        )
