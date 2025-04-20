import os
from classifier import logger
import pandas as pd
from classifier.entity.config_entity import TestDataCorrectionConfig
from classifier.Mylib import myfuncs
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


class TestDataCorrection:
    def __init__(self, config: TestDataCorrectionConfig):
        self.config = config

    def load_data(self):
        self.test_raw_data = myfuncs.load_python_object(self.config.test_raw_data_path)
        self.preprocessor = myfuncs.load_python_object(self.config.preprocessor_path)

    def transform_data(self):
        df_transformed = self.preprocessor.transform(self.test_raw_data)

        # Lưu dữ liệu
        myfuncs.save_python_object(self.config.test_data_path, df_transformed)
