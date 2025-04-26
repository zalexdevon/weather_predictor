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


class DC2:
    def __init__(self):
        pass

    def transform(self, df, data_type):
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


FEATURE_ORDINAL_DICT_DC2 = {}
