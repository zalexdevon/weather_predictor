import pandas as pd
import os
from classifier import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, GridSearchCV
from classifier.entity.config_entity import ModelTrainerConfig
from classifier.Mylib import myfuncs
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from xgboost import XGBClassifier
from scipy.stats import randint
import random
from lightgbm import LGBMClassifier
from sklearn.model_selection import ParameterSampler
from sklearn import metrics
from sklearn.base import clone
import time
from classifier.Mylib import myclasses
from classifier.Mylib import stringToObjectConverter
import timeit


class ManyModelsTypeModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data_to_train(self):
        # Load các training data
        self.train_feature_data = myfuncs.load_python_object(
            self.config.train_feature_path
        )
        self.train_target_data = myfuncs.load_python_object(
            self.config.train_target_path
        )
        self.val_feature_data = myfuncs.load_python_object(self.config.val_feature_path)
        self.val_target_data = myfuncs.load_python_object(self.config.val_target_path)

        # Load models
        self.models = [
            stringToObjectConverter.convert_string_to_object_4(model)
            for model in self.config.models
        ]

        self.num_models = len(self.models)

        # Load classes
        self.class_names = myfuncs.load_python_object(self.config.class_names_path)

    def train_model(self):
        print(
            f"\n========TIEN HANH TRAIN {self.num_models} MODELS !!!!!!================\n"
        )
        self.train_scorings = []
        self.val_scorings = []

        # Train model đầu tiên và get thời gian chạy ước tính
        print("Bắt đầu train model 0")
        first_model = self.models[0]
        start_time = time.time()
        first_model.fit(self.train_feature_data, self.train_target_data)

        train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            first_model,
            self.train_feature_data,
            self.train_target_data,
            self.config.scoring,
        )
        val_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            first_model,
            self.val_feature_data,
            self.val_target_data,
            self.config.scoring,
        )
        end_time = time.time()

        # In kết quả
        print(
            f"Model 0 -> Train {self.config.scoring}: {train_scoring}, Val {self.config.scoring}: {val_scoring}\n"
        )

        self.train_scorings.append(train_scoring)
        self.val_scorings.append(val_scoring)

        self.average_training_time = end_time - start_time
        self.estimated_all_models_train_time = (
            self.average_training_time * self.num_models
        )

        print(f"Thời gian trung bình chạy : {self.average_training_time} (min)")
        print(
            f"Thời gian ước tính chạy còn lại: {self.estimated_all_models_train_time} (min)"
        )

        for index, model in enumerate(self.models[1:]):
            print(f"Bắt đầu train model {index}")

            model.fit(self.train_feature_data, self.train_target_data)
            train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
                model,
                self.train_feature_data,
                self.train_target_data,
                self.config.scoring,
            )
            val_scoring = myfuncs.evaluate_model_on_one_scoring_17(
                model,
                self.val_feature_data,
                self.val_target_data,
                self.config.scoring,
            )

            # In kết quả
            print(
                f"Model {index} -> Train {self.config.scoring}: {train_scoring}, Val {self.config.scoring}: {val_scoring}\n"
            )

            self.train_scorings.append(train_scoring)
            self.val_scorings.append(val_scoring)

        print(
            f"\n========KET THUC TRAIN {self.num_models} MODELS !!!!!!================\n"
        )
        all_model_end_time = time.time()
        self.true_all_models_train_time = all_model_end_time - start_time
        self.true_average_train_time = self.true_all_models_train_time / self.num_models

    def save_best_model_results(self):
        # Tìm model tốt nhất và chỉ số train, val scoring tương ứng
        self.best_model, self.best_model_index, self.train_scoring, self.val_scoring = (
            myclasses.BestModelSearcher(
                self.models,
                self.train_scorings,
                self.val_scorings,
                self.config.target_score,
                self.config.scoring,
            ).next()
        )

        # Các chỉ số đánh giá của model
        self.best_model_results_text = "========KẾT QUẢ CỦA CÁC MODEL================\n"

        self.best_model_results_text += (
            f"Thời gian chạy trung bình cho 1 model: {self.true_average_train_time}\n"
        )
        self.best_model_results_text += (
            f"Thời gian chạy: {self.true_all_models_train_time}\n"
        )

        self.best_model_results_text += f"Chỉ số scoring của {self.num_models} model\n"
        for model_desc, train_scoring, val_scoring in zip(
            self.config.models, self.train_scorings, self.val_scorings
        ):
            self.best_model_results_text += f"{model_desc}\n-> train scoring: {train_scoring}, val scoring: {val_scoring}\n\n"

        self.best_model_results_text = (
            "========KẾT QUẢ CỦA BEST MODEL================\n"
        )
        self.best_model_results_text += "===THAM SỐ=====\n"
        self.best_model_results_text += f"{self.config.models[self.best_model_index]}"

        ## Chỉ số scoring
        self.best_model_results_text += f"\n\n====CHỈ SỐ SCORING====\n"
        self.best_model_results_text += (
            f"Train {self.config.scoring}: {self.train_scoring}\n"
        )
        self.best_model_results_text += (
            f"Val {self.config.scoring}: {self.val_scoring}\n"
        )

        # Các chỉ số khác bao gồm accuracy + classfication report + confusion matrix
        self.best_model_results_text += "====CÁC CHỈ SỐ KHÁC===========\n"
        best_model_results_text, train_confusion_matrix, val_confusion_matrix = (
            myclasses.ClassifierEvaluator(
                model=self.best_model,
                train_feature_data=self.train_feature_data,
                train_target_data=self.train_target_data,
                val_feature_data=self.val_feature_data,
                val_target_data=self.val_target_data,
                class_names=self.class_names,
            ).evaluate()
        )
        self.best_model_results_text += best_model_results_text

        train_confusion_matrix_path = os.path.join(
            self.config.root_dir, "train_confusion_matrix.png"
        )
        train_confusion_matrix.savefig(
            train_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
        )
        val_confusion_matrix_path = os.path.join(
            self.config.root_dir, "val_confusion_matrix.png"
        )
        val_confusion_matrix.savefig(
            val_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
        )

        # Lưu chỉ số đánh giá vào file results.txt
        with open(self.config.results_path, mode="w") as file:
            file.write(self.best_model_results_text)

        # Lưu lại model tốt nhất
        myfuncs.save_python_object(self.config.best_model_path, self.best_model)

    def save_list_monitor_components(self):
        self.train_scoring, self.val_scoring = myfuncs.get_value_with_the_meaning_28(
            (self.train_scoring, self.val_scoring), self.config.scoring
        )

        if os.path.exists(self.config.list_monitor_components_path):
            self.list_monitor_components = myfuncs.load_python_object(
                self.config.list_monitor_components_path
            )

        else:  # Tức đây là lần đầu training
            self.list_monitor_components = []

        self.list_monitor_components += [
            (
                self.config.model_name,
                self.train_scoring,
                self.val_scoring,
            )
        ]

        myfuncs.save_python_object(
            self.config.list_monitor_components_path, self.list_monitor_components
        )
