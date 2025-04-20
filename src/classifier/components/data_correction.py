import os
from classifier import logger
import pandas as pd
from classifier.entity.config_entity import DataCorrectionConfig
from classifier.Mylib import myfuncs
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


class UnnecessaryColsDeleter(
    BaseEstimator, TransformerMixin
):  # Tách ra nhiều giai đoạn
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        # Xoa cac cot khong can thiet
        df = df.drop(
            columns=[
                "ChestScan",
                "RaceEthnicityCategory",
                "BMI",
                "HIVTesting",
                "HighRiskLastYear",
                "CovidPos",
            ]
        )

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class ColNamesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        #  Đổi tên cột
        rename_dict = {
            "State": "State_nom",
            "Sex": "Sex_nom",
            "GeneralHealth": "GeneralHealth_ord",
            "PhysicalHealthDays": "PhysicalHealthDays_numcat",
            "MentalHealthDays": "MentalHealthDays_numcat",
            "LastCheckupTime": "LastCheckupTime_ord",
            "PhysicalActivities": "PhysicalActivities_bin",
            "SleepHours": "SleepHours_numcat",
            "RemovedTeeth": "RemovedTeeth_nom",
            "HadAngina": "HadAngina_bin",
            "HadStroke": "HadStroke_bin",
            "HadAsthma": "HadAsthma_bin",
            "HadSkinCancer": "HadSkinCancer_bin",
            "HadCOPD": "HadCOPD_bin",
            "HadDepressiveDisorder": "HadDepressiveDisorder_bin",
            "HadKidneyDisease": "HadKidneyDisease_bin",
            "HadArthritis": "HadArthritis_bin",
            "HadDiabetes": "HadDiabetes_nom",
            "DeafOrHardOfHearing": "DeafOrHardOfHearing_bin",
            "BlindOrVisionDifficulty": "BlindOrVisionDifficulty_bin",
            "DifficultyConcentrating": "DifficultyConcentrating_bin",
            "DifficultyWalking": "DifficultyWalking_bin",
            "DifficultyDressingBathing": "DifficultyDressingBathing_bin",
            "DifficultyErrands": "DifficultyErrands_bin",
            "SmokerStatus": "SmokerStatus_ord",
            "ECigaretteUsage": "ECigaretteUsage_ord",
            "AgeCategory": "AgeCategory_nom",
            "HeightInMeters": "HeightInMeters_num",
            "WeightInKilograms": "WeightInKilograms_num",
            "AlcoholDrinkers": "AlcoholDrinkers_bin",
            "FluVaxLast12": "FluVaxLast12_bin",
            "PneumoVaxEver": "PneumoVaxEver_bin",
            "TetanusLast10Tdap": "TetanusLast10Tdap_nom",
            "TetanusLast10Tdap": "TetanusLast10Tdap_nom",
            "HadHeartAttack": "HadHeartAttack_target",
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

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class ColDatatypePreprocessor(
    BaseEstimator, TransformerMixin
):  # Tách ra nhiều giai đoạn
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class NumericColDataPreprocessor(
    BaseEstimator, TransformerMixin
):  # Tách ra nhiều giai đoạn
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        # Cột HeightInMeters_num
        col_name = "HeightInMeters_num"
        df[col_name] = df[col_name].apply(lambda x: np.nan if x > 2.01 else x)

        # Cột WeightInKilograms_num
        col_name = "WeightInKilograms_num"
        df[col_name] = df[col_name].apply(lambda x: np.nan if x > 136.08 else x)

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class NumericCatColDataPreprocessor(
    BaseEstimator, TransformerMixin
):  # Tách ra nhiều giai đoạn
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        # UPDATE Cột SleepHours_num
        col_name = "SleepHours_numcat"
        replaced_value = list(range(13, 24 + 1))
        replaced_dict = dict(zip(replaced_value, [np.nan] * len(replaced_value)))
        df[col_name] = df[col_name].replace(replaced_dict)

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class BinaryColDataPreprocessor(
    BaseEstimator, TransformerMixin
):  # Tách ra nhiều giai đoạn
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class NominalColDataPreprocessor(
    BaseEstimator, TransformerMixin
):  # Tách ra nhiều giai đoạn
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class OrdinalColDataPreprocessor(
    BaseEstimator, TransformerMixin
):  # Tách ra nhiều giai đoạn
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class TargetColDataPreprocessor(
    BaseEstimator, TransformerMixin
):  # Tách ra nhiều giai đoạn
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class HandleMissingValuePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        numeric_cols, numericCat_cols, cat_cols, _, _, _, target_col = (
            myfuncs.get_different_types_cols_from_df_4(X)
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="mean"), numeric_cols),
                ("numCat", SimpleImputer(strategy="most_frequent"), numericCat_cols),
                ("cat", SimpleImputer(strategy="most_frequent"), cat_cols),
                ("target", SimpleImputer(strategy="most_frequent"), [target_col]),
            ]
        )

        self.preprocessor.fit(X)

        return self

    def transform(self, X, y=None):
        X = self.preprocessor.transform(X)
        return pd.DataFrame(
            X,
            columns=myfuncs.get_real_column_name_from_get_feature_names_out(
                self.preprocessor.get_feature_names_out()
            ),
        )

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class AfterHandleMissingValuePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        numeric_cols, numericCat_cols, cat_cols, binary_cols, _, _, target_col = (
            myfuncs.get_different_types_cols_from_df_4(df)
        )

        # Chuyen doi ve dung kieu du lieu
        df[numeric_cols] = df[numeric_cols].astype("float")
        df[numericCat_cols] = df[numericCat_cols].astype("float")
        df[cat_cols + [target_col]] = df[cat_cols + [target_col]].astype("category")

        # Thay doi thu tu cac label cho cac cot ordinal, binary va target
        bin_values_dict = dict(zip(binary_cols, [["No", "Yes"]] * len(binary_cols)))

        for col, value in bin_values_dict.items():
            df[col] = df[col].cat.reorder_categories(value, ordered=True)

        ord_values_dict = {
            "GeneralHealth_ord": ["Poor", "Fair", "Good", "Very good", "Excellent"],
            "LastCheckupTime_ord": [
                "5 or more years ago",
                "Within past 5 years (2 years but less than 5 years ago)",
                "Within past 2 years (1 year but less than 2 years ago)",
                "Within past year (anytime less than 12 months ago)",
            ],
            "SmokerStatus_ord": [
                "Current smoker - now smokes every day",
                "Current smoker - now smokes some days",
                "Former smoker",
                "Never smoked",
            ],
            "ECigaretteUsage_ord": [
                "Use them every day",
                "Use them some days",
                "Not at all (right now)",
                "Never used e-cigarettes in my entire life",
            ],
        }

        for col, value in ord_values_dict.items():
            df[col] = df[col].cat.reorder_categories(value, ordered=True)

        df[target_col] = df[target_col].cat.reorder_categories(
            ["No", "Yes"], ordered=True
        )

        # Loại bỏ duplicates
        df = df.drop_duplicates()

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class DataCorrection:
    def __init__(self, config: DataCorrectionConfig):
        self.config = config

    def load_data(self):
        self.train_raw_data = myfuncs.load_python_object(
            self.config.train_raw_data_path
        )

    def create_preprocessor_for_train_data(self):
        self.preprocessor = Pipeline(
            steps=[
                ("1", UnnecessaryColsDeleter()),
                ("2", ColNamesPreprocessor()),
                ("3", ColDatatypePreprocessor()),
                ("4", NumericColDataPreprocessor()),
                ("5", NumericCatColDataPreprocessor()),
                ("6", BinaryColDataPreprocessor()),
                ("7", NominalColDataPreprocessor()),
                ("8", OrdinalColDataPreprocessor()),
                ("9", TargetColDataPreprocessor()),
                ("during", HandleMissingValuePreprocessor()),
                ("after", AfterHandleMissingValuePreprocessor()),
            ]
        )

    def transform_data(self):
        df_transformed = self.preprocessor.fit_transform(
            self.train_raw_data
        ).reset_index(drop=True)

        target_col = myfuncs.get_target_col_from_df_26(df_transformed)

        # Chia thành tập train, val
        df_train, df_val = train_test_split(
            df_transformed,
            test_size=self.config.val_size,
            stratify=df_transformed[target_col],
            random_state=42,
        )

        # Get class_names
        class_names = df_transformed[target_col].cat.categories.tolist()

        # Lưu dữ liệu
        myfuncs.save_python_object(self.config.preprocessor_path, self.preprocessor)
        myfuncs.save_python_object(self.config.train_data_path, df_train)
        myfuncs.save_python_object(self.config.val_data_path, df_val)
        myfuncs.save_python_object(self.config.class_names_path, class_names)
