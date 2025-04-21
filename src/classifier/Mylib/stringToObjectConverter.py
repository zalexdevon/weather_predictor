import ast
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from classifier.Mylib.myclasses import (
    ConvNetBlock_XceptionVersion,
    ConvNetBlock_Advanced,
    ConvNetBlock,
    ImageDataPositionAugmentation,
    ImageDataColorAugmentation,
    PretrainedModel,
    ManyConvNetBlocks_XceptionVersion,
    ManyConvNetBlocks_Advanced,
    ManyConvNetBlocks,
)
from tensorflow.keras.layers import (
    Resizing,
    Rescaling,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input

from tensorflow.keras.optimizers import RMSprop
from sklearn.linear_model import LogisticRegression
from classifier.Mylib.myclasses import ColumnsDeleter
from sklearn.decomposition import PCA


def do_ast_literal_eval_advanced_7(text: str):
    """Kế thừa hàm ast.literal_eval() nhưng xử lí thêm trường hợp sau

    Tuple, List dạng (1.0 ; 2.0), các phần tử cách nhau bởi dấu ; thay vì dấu ,

    """
    if ";" not in text:
        return ast.literal_eval(text)

    return ast.literal_eval(text.replace(";", ","))


def convert_string_to_object_4(text: str):
    """Chuyển 1 chuỗi thành 1 đối tượng

    Example:
        text = "LogisticRegression(C=144, penalty=l1, solver=saga,max_iter=10000,dual=True)"

        -> đối tượng LogisticRegression(C=144, dual=True, max_iter=10000, penalty='l1',solver='saga')

    Args:
        text (str): _description_


    """
    # Tách tên lớp và tham số
    class_name, params = text.split("(", 1)
    params = params[:-1]

    object_class = globals()[class_name]

    if params == "":
        return object_class()

    # Lấy tham số của đối tượng
    param_parts = params.split(",")
    param_parts = [item.strip() for item in param_parts]
    keys = [item.split("=")[0].strip() for item in param_parts]

    values = [
        do_ast_literal_eval_advanced_7(item.strip().split("=")[1].strip())
        for item in param_parts
    ]

    params = dict(zip(keys, values))

    return object_class(**params)
