from classifier.config.configuration import ConfigurationManager
from classifier.components.data_correction import DataCorrection
from classifier import logger
from pathlib import Path
import traceback

STAGE_NAME = "Data Correction stage"


class DataCorrectionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_correction_config = config.get_data_correction_config()

        data_corrrection = DataCorrection(config=data_correction_config)

        try:
            data_corrrection.load_data()
            print("\n===== Load data thành công ====== \n")

            data_corrrection.create_preprocessor_for_train_data()
            print("\n===== Tạo preprocessor thành công ====== \n")

            data_corrrection.transform_data()
            print("\n===== Transform data thành công ====== \n")

            print("================ NO ERORR :)))))))))) ==========================")
        except Exception as e:
            print(f"==========ERROR: =============")
            print(f"Exception: {e}\n")
            print("=====Traceback========\n")
            traceback.print_exc()
            exit(1)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_transform = DataCorrectionPipeline()
        data_transform.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
