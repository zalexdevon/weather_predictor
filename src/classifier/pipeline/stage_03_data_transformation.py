from classifier.config.configuration import ConfigurationManager
from classifier.components.data_transformation import DataTransformation
from classifier import logger
from pathlib import Path
import traceback

STAGE_NAME = "Data Transformation stage"


class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)

        try:
            data_transformation.load_data()
            print("\n===== Load data thành công ====== \n")

            data_transformation.create_preprocessor_for_train_data()
            print("\n===== Tạo preprocessor thành công ====== \n")

            data_transformation.transform_data()
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
        data_transform = DataTransformationPipeline()
        data_transform.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
