from classifier.constants import *
from Mylib.myfuncs import read_yaml, create_directories
from classifier.entity.config_entity import (
    DataCorrectionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    MonitorPlotterConfig,
    TestDataCorrectionConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_correction_config(self) -> DataCorrectionConfig:
        config = self.config.data_correction
        params = self.params.data_correction

        create_directories([config.root_dir])

        return DataCorrectionConfig(
            # config input
            train_raw_data_path=config.train_raw_data_path,
            # config output
            root_dir=config.root_dir,
            preprocessor_path=config.preprocessor_path,
            data_path=config.data_path,
            train_data_path=config.train_data_path,
            val_data_path=config.val_data_path,
            class_names_path=config.class_names_path,
            # params
            val_size=params.val_size,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            # config input
            train_data_path=config.train_data_path,
            val_data_path=config.val_data_path,
            # config output
            root_dir=config.root_dir,
            preprocessor_path=config.preprocessor_path,
            train_features_path=config.train_features_path,
            train_target_path=config.train_target_path,
            val_features_path=config.val_features_path,
            val_target_path=config.val_target_path,
            # params
            do_smote=params.do_smote,
            list_after_feature_transformer=params.list_after_feature_transformer,
        )

        return data_transformation_config

    def get_model_trainer_config(
        self,
    ) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            # config input
            train_feature_path=config.train_feature_path,
            train_target_path=config.train_target_path,
            val_feature_path=config.val_feature_path,
            val_target_path=config.val_target_path,
            class_names_path=config.class_names_path,
            # config output
            root_dir=config.root_dir,
            best_model_path=config.best_model_path,
            results_path=config.results_path,
            list_monitor_components_path=config.list_monitor_components_path,
            # params
            model_name=params.model_name,
            model_training_type=params.model_training_type,
            # params to use GridSearch
            base_model=params.base_model,
            n_iter=params.n_iter,
            param_grid=params.param_grid,
            # params to train many models
            models=params.models,
            # common params
            scoring=self.params.scoring,
            target_score=self.params.target_score,
        )

        return model_trainer_config

    # TEST DATA CORRECTION
    def get_test_data_correction_config(self) -> TestDataCorrectionConfig:
        config = self.config.test_data_correction

        create_directories([config.root_dir])

        obj = TestDataCorrectionConfig(
            # input
            test_raw_data_path=config.test_raw_data_path,
            preprocessor_path=config.preprocessor_path,
            # output
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
        )

        return obj

    # MODEL_EVALUATION
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        obj = ModelEvaluationConfig(
            # input
            test_data_path=config.test_data_path,
            preprocessor_path=config.preprocessor_path,
            model_path=config.model_path,
            class_names_path=config.class_names_path,
            # output
            root_dir=config.root_dir,
            results_path=config.results_path,
            # common params
            scoring=self.params.scoring,
        )

        return obj

    def get_monitor_plot_config(self) -> MonitorPlotterConfig:
        config = self.params.monitor_plotter

        obj = MonitorPlotterConfig(
            monitor_plot_html_path=config.monitor_plot_html_path,
            monitor_plot_fig_path=config.monitor_plot_fig_path,
            target_val_value=config.target_val_value,
            max_val_value=config.max_val_value,
            dtick_y_value=config.dtick_y_value,
        )

        return obj
