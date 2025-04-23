import pandas as pd
import os
from classifier import logger
from classifier.entity.config_entity import ModelEvaluationConfig
from Mylib import myfuncs
from sklearn import metrics
from Mylib import myclasses


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_model(self):
        # Load data
        df = myfuncs.load_python_object(self.config.test_data_path)
        preprocessor = myfuncs.load_python_object(self.config.preprocessor_path)
        self.model = myfuncs.load_python_object(self.config.model_path)
        self.class_names = myfuncs.load_python_object(self.config.class_names_path)

        # Transform test data
        target_col = myfuncs.get_target_col_from_df_26(df_transformed)
        df_transformed = preprocessor.transform(df)
        self.df_feature = df_transformed.drop(columns=[target_col])
        self.df_target = df_transformed[target_col]

        # Các chỉ số đánh giá của model
        self.model_results_text = "========KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH================\n"

        ## Chỉ số scoring
        test_score = myfuncs.evaluate_model_on_one_scoring_17(
            self.model, self.df_feature, self.df_target, self.config.scoring
        )
        self.model_results_text += f"====CHỈ SỐ SCORING====\n"
        self.model_results_text += f"{self.config.scoring}: {test_score}"

        # Các chỉ số khác
        self.model_results_text += "====CÁC CHỈ SỐ KHÁC===========\n"
        model_results_text, test_confusion_matrix = myclasses.ClassifierEvaluator(
            model=self.model,
            class_names=self.class_names,
            train_feature_data=self.df_feature,
            train_target_data=self.df_target,
        ).evaluate()
        self.model_results_text += model_results_text

        test_confusion_matrix_path = os.path.join(
            self.config.root_dir, "test_confusion_matrix.png"
        )
        test_confusion_matrix.savefig(
            test_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
        )

        # Lưu chỉ số đánh giá vào file results.txt
        with open(self.config.results_path, mode="w") as file:
            file.write(self.model_results_text)
