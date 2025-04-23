import os
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from classifier.entity.config_entity import ModelTrainerConfig
from Mylib import myfuncs
from sklearn.base import clone
from Mylib import myclasses
from Mylib import stringToObjectConverter


class ModelTrainer:
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

        # Gộp train, val vào đúng 1 df
        self.features, self.target, self.trainval_splitter = (
            myfuncs.get_features_target_spliter_for_CV_train_val(
                self.train_feature_data,
                self.train_target_data,
                self.val_feature_data,
                self.val_target_data,
            )
        )

        # Load base model (chưa có tham số)
        self.base_model = (
            stringToObjectConverter.convert_complex_MLmodel_yaml_to_object(
                self.config.base_model
            )
        )

        # Load params thực hiện fine tune model
        self.param_grid = myfuncs.get_param_grid_model(self.config.param_grid)

        # Load scoring để sử dụng vào RandomizedSearchCV, GridSearchCV (vd: log_loss -> neg_log_loss)
        if self.config.scoring == "accuracy":
            self.scoring = "accuracy"
        elif self.config.scoring == "log_loss":
            self.scoring = "neg_log_loss"
        elif self.config.scoring == "mse":
            self.scoring = "neg_mean_squared_error"
        elif self.config.scoring == "mae":
            self.scoring = "neg_mean_absolute_error"
        else:
            raise ValueError("Chỉ mới định nghĩa cho accuracy, log_loss, mse, mae")

        # Load searcher
        if self.config.model_training_type == "r":
            self.searcher = RandomizedSearchCV(
                self.base_model,
                param_distributions=self.param_grid,
                n_iter=self.config.n_iter,
                cv=self.trainval_splitter,
                random_state=42,
                scoring=self.scoring,
                return_train_score=True,
                verbose=2,
            )
        elif self.config.model_training_type == "g":
            self.searcher = GridSearchCV(
                self.base_model,
                param_grid=self.param_grid,
                cv=self.trainval_splitter,
                scoring=self.scoring,
                return_train_score=True,
                verbose=2,
            )
        elif self.config.model_training_type == "rcv":
            self.searcher = RandomizedSearchCV(
                self.base_model,
                param_distributions=self.param_grid,
                n_iter=self.config.n_iter,
                cv=5,
                random_state=42,
                scoring=self.scoring,
                return_train_score=True,
                verbose=2,
            )
        elif self.config.model_training_type == "gcv":
            self.searcher = GridSearchCV(
                self.base_model,
                param_grid=self.param_grid,
                cv=5,
                scoring=self.scoring,
                return_train_score=True,
                verbose=2,
            )
        else:
            raise ValueError(
                "===== Giá trị model_training_type không hợp lệ =============="
            )

        # Load classes
        self.class_names = myfuncs.load_python_object(self.config.class_names_path)

    def train_model(self):
        print(f"\n========TIEN HANH TRAIN MODELS !!!!!!================\n")

        self.searcher.fit(self.features, self.target)

        print(f"\n========KET THUC TRAIN MODELS !!!!!!================\n")

    def save_best_model_results(self):
        # Tìm model tốt nhất và chỉ số đánh giá tương ứng
        self.best_model = self.searcher.best_estimator_

        self.cv_results = self.searcher.cv_results_

        self.train_scoring, self.val_scoring = (
            myfuncs.find_best_model_train_val_scoring_when_using_RandomisedSearch_GridSearch_19(
                self.cv_results, self.config.scoring
            )
        )

        # Các chỉ số đánh giá của model
        self.best_model_results_text = (
            "========KET QUA CUA MO HINH TOT NHAT================\n"
        )

        self.best_model_results_text += "===THAM SỐ=====\n"
        self.best_model_results_text += self.searcher.best_params_

        ## Chỉ số scoring
        self.best_model_results_text += f"====CHỈ SỐ SCORING====\n"
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

        else:
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
