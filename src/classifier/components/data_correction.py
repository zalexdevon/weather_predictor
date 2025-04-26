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
from classifier.data_correction_code.dc1 import DC1, FEATURE_ORDINAL_DICT_DC1
from classifier.data_correction_code.dc2 import DC2, FEATURE_ORDINAL_DICT_DC2


class DataCorrection:
    def __init__(self, config: DataCorrectionConfig):
        self.config = config

    def load_data(self):
        self.df = myfuncs.load_python_object(self.config.train_data_path)

    def create_preprocessor_for_train_data(self):
        self.transformer = DC2()

    def transform_data(self):
        df = self.transformer.transform(self.df, data_type="train")

        myfuncs.save_python_object(self.config.data_path, df)
        myfuncs.save_python_object(
            self.config.feature_ordinal_dict_path, FEATURE_ORDINAL_DICT_DC2
        )
        myfuncs.save_python_object(
            self.config.correction_transformer_path, self.transformer
        )
