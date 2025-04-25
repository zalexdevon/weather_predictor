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


class DataCorrectorForTrainAndTest:
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
            "sunrise": "sunrise_ord",
            "sunset": "sunset_ord",
            "moonrise": "moonrise_ord",
            "moonset": "moonset_ord",
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

        # Kiểm tra nội dung cột Ordinal
        ## Biển đổi 4 cột sunrise_ord, sunset_ord, moonrise_ord, moonset_ord
        format = r"\d+:\d+\s*(AM|PM)"

        df_ordinal_cols = df[ordinal_cols]
        index_not_satisfy_format = (
            df_ordinal_cols[
                df_ordinal_cols.applymap(
                    lambda item: re.fullmatch(format, item) is None
                )
            ]
            .stack()
            .index
        )

        df_ordinal_cols_happen_stack = df_ordinal_cols.stack()
        df_ordinal_cols_happen_stack = df_ordinal_cols_happen_stack[
            ~df_ordinal_cols_happen_stack.index.isin(index_not_satisfy_format)
        ]
        df_ordinal_cols_happen_stack = df_ordinal_cols_happen_stack.apply(
            lambda item: process_time(item)
        )

        df_ordinal_cols_stack = df_ordinal_cols.stack()
        df_ordinal_cols_stack[df_ordinal_cols_happen_stack.index] = (
            df_ordinal_cols_happen_stack
        )
        df[ordinal_cols] = df_ordinal_cols_stack.unstack()

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


FEATURE_ORDINAL_DICT = {
    "sunrise_ord": ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"],
    "sunset_ord": ["14", "15", "16", "17", "18", "19", "20", "21", "22", "23"],
    "moonrise_ord": [
        "No moonrise",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
    ],
    "moonset_ord": [
        "No moonset",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
    ],
}


class DataCorrection:
    def __init__(self, config: DataCorrectionConfig):
        self.config = config

    def load_data(self):
        self.df = myfuncs.load_python_object(self.config.train_data_path)

    def create_preprocessor_for_train_data(self):
        self.transformer = DataCorrectorForTrainAndTest()

    def transform_data(self):
        df = self.transformer.transform(self.df, data_type="train")

        myfuncs.save_python_object(self.config.data_path, df)
        myfuncs.save_python_object(
            self.config.feature_ordinal_dict_path, FEATURE_ORDINAL_DICT
        )
        myfuncs.save_python_object(
            self.config.correction_transformer_path, self.transformer
        )
