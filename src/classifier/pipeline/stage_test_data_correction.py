from classifier.config.configuration import ConfigurationManager
from classifier.components.test_data_correction import TestDataCorrection
from classifier import logger
from pathlib import Path
import traceback

STAGE_NAME = "Test Data Correction stage"


class TestDataCorrectionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_test_data_correction_config()
        data_transformation = TestDataCorrection(config=data_transformation_config)

        try:
            data_transformation.load_data()
            print("\n===== Load test data thành công ====== \n")

            data_transformation.transform_data()
            print("\n===== Transform test data thành công ====== \n")

            print("================ NO ERORR :)))))))))) ==========================")
        except Exception as e:
            print(f"==========ERROR: =============")
            print(f"Exception: {e}\n")
            print("=====Traceback========\n")
            traceback.print_exc()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_transform = TestDataCorrectionPipeline()
        data_transform.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
