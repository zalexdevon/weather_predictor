from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataCorrectionConfig:
    # config input
    train_raw_data_path: Path

    # config output
    root_dir: Path
    preprocessor_path: Path
    train_data_path: Path
    val_data_path: Path
    class_names_path: Path

    # params
    val_size: float


@dataclass(frozen=True)
class DataTransformationConfig:
    # config input
    train_data_path: Path
    val_data_path: Path

    # config output
    root_dir: Path
    preprocessor_path: Path
    train_features_path: Path
    train_target_path: Path
    val_features_path: Path
    val_target_path: Path

    # params
    do_smote: str
    list_after_feature_transformer: list


@dataclass(frozen=True)
class ModelTrainerConfig:
    # config input
    train_feature_path: Path
    train_target_path: Path
    val_feature_path: Path
    val_target_path: Path
    class_names_path: Path

    # config output
    root_dir: Path
    best_model_path: Path
    results_path: Path
    list_monitor_components_path: Path

    # params
    model_name: str
    model_training_type: str

    # params to use GridSearch
    base_model: str
    n_iter: int
    param_grid: dict

    # params to train many models
    models: list

    # common params
    scoring: str
    target_score: float


# TEST DATA CORRECTION
@dataclass(frozen=True)
class TestDataCorrectionConfig:
    # input
    test_raw_data_path: Path
    preprocessor_path: Path

    # output
    root_dir: Path
    test_data_path: Path


# MODEL_EVALUATION
@dataclass(frozen=True)
class ModelEvaluationConfig:
    # input
    test_data_path: Path
    preprocessor_path: Path
    model_path: Path
    class_names_path: Path

    # output
    root_dir: Path
    results_path: Path

    # common params
    scoring: str


@dataclass(frozen=True)
class MonitorPlotterConfig:
    monitor_plot_html_path: Path
    monitor_plot_fig_path: Path
    target_val_value: float
    max_val_value: float
    dtick_y_value: float
