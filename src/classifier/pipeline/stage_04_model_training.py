from classifier.config.configuration import ConfigurationManager
from classifier.components.model_trainer import (
    ModelTrainer,
)
from classifier.components.many_models_type_model_trainer import (
    ManyModelsTypeModelTrainer,
)
from classifier import logger
from classifier.components.monitor_plotter import (
    MonitorPlotter,
)
import traceback

STAGE_NAME = "Model Trainer stage"


#
class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()

        model_trainer = None
        if model_trainer_config.model_training_type == "m":
            model_trainer = ManyModelsTypeModelTrainer(model_trainer_config)
        else:
            model_trainer = ModelTrainer(model_trainer_config)

        try:
            model_trainer.load_data_to_train()
            print("\n===== Load data thành công ====== \n")

            model_trainer.train_model()
            print("\n===== Train data thành công ====== \n")

            model_trainer.save_best_model_results()
            print("\n===== Save kết quả đánh giá best model thành công ====== \n")

            model_trainer.save_list_monitor_components()
            monitor_plot_config = config.get_monitor_plot_config()
            monitor_plot = MonitorPlotter(config=monitor_plot_config)
            monitor_plot.plot(model_trainer.list_monitor_components)
            print("\n===== Save kết quả các lần chạy model thành công ====== \n")

            print("================ NO ERORR :)))))))))) ==========================")
        except Exception as e:
            print(f"==========ERROR: =============")
            print(f"Exception: {e}\n")
            print("=====Traceback========")
            traceback.print_exc()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
