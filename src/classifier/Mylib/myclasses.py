import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import keras_cv
import matplotlib.cm as cm
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from classifier.Mylib import myfuncs
from sklearn import metrics


class ConvNetBlock_XceptionVersion(layers.Layer):
    """Gồm các layers sau:
    - SeparableConv2D
    - SeparableConv2D
    - MaxPooling2D

    Đi kèm là sự kết hợp giữa residual connections, batch normalization và  depthwise separable convolutions (lớp SeparableConv2D)

    Attributes:
        filters (_type_): số lượng filters trong lớp SeparableConv2D
    """

    def __init__(self, filters, name=None, **kwargs):
        """_summary_"""
        # super(ConvNetBlock_XceptionVersion, self).__init__()
        super().__init__(name=name, **kwargs)
        self.filters = filters

    def build(self, input_shape):

        self.BatchNormalization = layers.BatchNormalization()
        self.Activation = layers.Activation("relu")
        self.SeparableConv2D = layers.SeparableConv2D(
            self.filters, 3, padding="same", use_bias=False
        )

        self.BatchNormalization_1 = layers.BatchNormalization()
        self.Activation_1 = layers.Activation("relu")
        self.SeparableConv2D_1 = layers.SeparableConv2D(
            self.filters, 3, padding="same", use_bias=False
        )
        self.MaxPooling2D = layers.MaxPooling2D(3, strides=2, padding="same")

        self.Conv2D = layers.Conv2D(
            self.filters, 1, strides=2, padding="same", use_bias=False
        )

        super().build(input_shape)

    def call(self, x):
        residual = x

        # First part of the block
        x = self.BatchNormalization(x)
        x = self.Activation(x)
        x = self.SeparableConv2D(x)

        # Second part of the block
        x = self.BatchNormalization_1(x)
        x = self.Activation_1(x)
        x = self.SeparableConv2D_1(x)
        x = self.MaxPooling2D(x)

        # Apply residual connection
        residual = self.Conv2D(residual)
        x = layers.add([x, residual])

        return x

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        name = config.pop("name", None)
        return cls(**config)


class ConvNetBlock_Advanced(layers.Layer):
    """Gồm các layers sau:
    - Conv2D
    - Conv2D
    - MaxPooling2D

    Đi kèm là sự kết hợp giữa residual connections, batch normalization

    Attributes:
        filters (_type_): số lượng filters trong lớp Conv2D
    """

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):

        self.BatchNormalization = layers.BatchNormalization()
        self.Activation = layers.Activation("relu")
        self.Conv2D = layers.Conv2D(self.filters, 3, padding="same")

        self.BatchNormalization_1 = layers.BatchNormalization()
        self.Activation_1 = layers.Activation("relu")
        self.Conv2D_1 = layers.Conv2D(self.filters, 3, padding="same")
        self.MaxPooling2D = layers.MaxPooling2D(2, padding="same")

        self.Conv2D_2 = layers.Conv2D(self.filters, 1, strides=2)

        super().build(input_shape)

    def call(self, x):
        residual = x

        # First part of the block
        x = self.BatchNormalization(x)
        x = self.Activation(x)
        x = self.Conv2D(x)

        # Second part of the block
        x = self.BatchNormalization_1(x)
        x = self.Activation_1(x)
        x = self.Conv2D_1(x)
        x = self.MaxPooling2D(x)

        # Xử lí residual
        residual = self.Conv2D_2(residual)

        # Apply residual connection
        x = layers.add([x, residual])

        return x

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class ConvNetBlock(layers.Layer):
    """Kết hợp các layers sau:
    - Conv2D * num_Conv2D_layer
    - MaxPooling

    Attributes:
        filters (_type_): số lượng filters trong lớp Conv2D
        num_Conv2D (int, optional): số lượng lớp num_Conv2D. Defaults to 1.
    """

    def __init__(self, filters, num_Conv2D=1, name=None, **kwargs):
        """ """
        # super(ConvNetBlock, self).__init__()
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.num_Conv2D = num_Conv2D

    def build(self, input_shape):
        self.list_Conv2D = [
            layers.Conv2D(self.filters, 3, activation="relu")
            for _ in range(self.num_Conv2D)
        ]

        self.MaxPooling2D = layers.MaxPooling2D(pool_size=2)

        super().build(input_shape)

    def call(self, x):
        for conv2D in self.list_Conv2D:
            x = conv2D(x)

        x = self.MaxPooling2D(x)

        return x

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "num_Conv2D": self.num_Conv2D,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ImageDataPositionAugmentation(layers.Layer):
    """Tăng cường dữ liệu hình ảnh ở khía cạnh vị trí, bao gồm các lớp sau (**trong tf.keras.layers**)
    - RandomFlip
    - RandomRotation
    - RandomZoom

    Attributes:
        rotation_factor (float): Tham số cho lớp RandomRotation. Default to 0.2
        zoom_factor (float): Tham số cho lớp RandomZoom. Default to 0.2
    """

    def __init__(self, rotation_factor=0.2, zoom_factor=0.2, **kwargs):
        # super(ImageDataPositionAugmentation, self).__init__()
        super().__init__(**kwargs)
        self.rotation_factor = rotation_factor
        self.zoom_factor = zoom_factor

    def build(self, input_shape):
        self.RandomFlip = layers.RandomFlip(mode="horizontal_and_vertical")
        self.RandomRotation = layers.RandomRotation(factor=self.rotation_factor)
        self.RandomZoom = layers.RandomZoom(height_factor=self.zoom_factor)

        super().build(input_shape)

    def call(self, x):
        x = self.RandomFlip(x)
        x = self.RandomRotation(x)
        x = self.RandomZoom(x)

        return x

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "rotation_factor": self.rotation_factor,
                "zoom_factor": self.zoom_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        name = config.pop("name", None)
        return cls(**config)


class ImageDataColorAugmentation(layers.Layer):
    """Tăng cường dữ liệu hình ảnh ở khía cạnh màu sắc, bao gồm các lớp sau (**trong keras_cv.layers**)
    - RandomBrightness
    - RandomGaussianBlur
    - RandomContrast
    - RandomHue
    - RandomSaturation

    Attributes:
        brightness_factor (float, optional): factor cho RandomBrightness. Defaults to 0.2.
        contrast_factor (float, optional): factor cho RandomContrast. Defaults to 0.2.
        hue_factor (float, optional): factor cho RandomHue. Defaults to 0.2.
        saturation_factor (float, optional): factor cho RandomSaturation. Defaults to 0.2.
    """

    def __init__(
        self,
        brightness_factor=0.2,
        contrast_factor=0.2,
        hue_factor=0.2,
        saturation_factor=0.2,
        **kwargs,
    ):
        # super(ImageDataColorAugmentation, self).__init__()
        super().__init__(**kwargs)
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.hue_factor = hue_factor
        self.saturation_factor = saturation_factor

    def build(self, input_shape):
        self.RandomBrightness = keras_cv.layers.RandomBrightness(
            factor=self.brightness_factor
        )
        self.RandomGaussianBlur = keras_cv.layers.RandomGaussianBlur(
            kernel_size=3, factor=(0.0, 1.0)
        )
        self.RandomContrast = keras_cv.layers.RandomContrast(
            factor=self.contrast_factor,
            value_range=(1 - self.contrast_factor, 1 + self.contrast_factor),
        )
        self.RandomHue = keras_cv.layers.RandomHue(
            factor=self.hue_factor,
            value_range=(1 - self.hue_factor, 1 + self.hue_factor),
        )
        self.RandomSaturation = keras_cv.layers.RandomSaturation(
            factor=self.saturation_factor
        )

        super().build(input_shape)

    def call(self, x):
        x = self.RandomBrightness(x)
        x = self.RandomGaussianBlur(x)  # Lớp này để mặc định
        x = self.RandomContrast(x)
        x = self.RandomHue(x)
        x = self.RandomSaturation(x)

        return x

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh, bao gồm cả tham số trainable và dtype
        config = super().get_config()
        config.update(
            {
                "brightness_factor": self.brightness_factor,
                "contrast_factor": self.contrast_factor,
                "hue_factor": self.hue_factor,
                "saturation_factor": self.saturation_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Loại bỏ tham số 'name' từ config (vì Keras đã xử lý nó)
        name = config.pop("name", None)
        return cls(**config)


class PretrainedModel(layers.Layer):
    """Sử dụng các pretrained models ở trong **keras.applications**
    Attributes:
        model_name (str): Tên pretrained model, vd: vgg16, vgg19, ....
        num_trainable (int, optional): Số lượng các lớp đầu tiên cho trainable = True. Defaults to 0.
    """

    def __init__(self, model_name, num_trainable=0, **kwargs):
        if num_trainable < 0:
            raise ValueError(
                "=========ERROR: Tham số <num_trainable> trong class PretrainedModel phải >= 0   ============="
            )

        # super(ConvNetBlock, self).__init__()
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_trainable = num_trainable

    def build(self, input_shape):
        if self.model_name == "vgg16":
            self.model = keras.applications.vgg16.VGG16(
                weights="imagenet", include_top=False
            )
            self.preprocess_input = keras.applications.vgg16.preprocess_input
        elif self.model_name == "vgg19":
            self.model = keras.applications.vgg19.VGG19(
                weights="imagenet", include_top=False
            )
            self.preprocess_input = keras.applications.vgg19.preprocess_input
        else:
            raise ValueError(
                "=========ERROR: Pretrained model name is not valid============="
            )

        # Cập nhật trạng thái trainable cho các lớp đầu
        if self.num_trainable == 0:
            self.model.trainable = False
        else:
            self.model.trainable = True
            for layer in self.model.layers[: -self.num_trainable]:
                layer.trainable = False

        super().build(input_shape)

    def call(self, x):
        x = self.preprocess_input(x)
        x = self.model(x)

        return x


class GradCAMForImages:
    """Thực hiện quá trình GradCAM để xác định những phần nảo của ảnh hỗ trợ model phân loại nhiều nhất
    Attributes:
        images (np.ndarray): Tập ảnh đã được chuyển thành **array**
        model (_type_): model
        last_convnet_layer_name ([str, int]): **Tên** hoặc  **index** của layer convent cuối cùng trong model

    Hàm -> **convert()**

    Returns:
        list_superimposed_img (list[PIL.Image.Image]): 1 mảng

    Examples:
        Nhấn mạnh những phần trên ảnh giúp phân loại các lá -> 3 loại: healthy, early_blight, late_blight
        ```python
        # Lấy đường dẫn của các ảnh
        img_paths = [os.path.join(folder, file) for file in file_names]

        # Chuyển các ảnh thành các mảng numpy
        file_names_array = myclasses.ImagesToArrayConverter(image_paths=img_paths, target_size=256).convert()

        # Load model
        model = load_model("artifacts/model_trainer/CONVNET_45/best_model.keras")
        last_convnet_index = int(3) # Specify lớp convnet cuối cùng (thông qua chỉ số), mà cũng nên dùng chỉ số đi :))))

        # Kết quả thu được là 1 mảng các PIL.Image.Image
        result = myclasses.GradCAMForImages(file_names_array, model, last_convnet_index).convert()

        # Show các ảnh lên
        for image in result:
            plt.imshow(image)
        ```
    """

    def __init__(self, images, model, last_convnet_layer_name):
        self.images = images
        self.model = model
        self.last_convnet_layer_name = last_convnet_layer_name

    def create_models(self):
        """Tạo ra 2 model sau:

        **last_conv_layer_model**: model map input image -> convnet block cuối cùng

        **classifier_model**: model map convnet block cuối cùng -> final class predictions.

        Returns:
            tuple: last_conv_layer_model, classifier_model
        """
        last_conv_layer = None
        classifier_layers = None
        if isinstance(self.last_convnet_layer_name, str):
            layer_names = [layer.name for layer in self.model.layers]
            last_conv_layer = self.model.get_layer(self.last_convnet_layer_name)
            classifier_layers = self.model.layers[
                layer_names.index(self.last_convnet_layer_name) + 1 :
            ]
        else:
            last_conv_layer = self.model.layers[self.last_convnet_layer_name]
            classifier_layers = self.model.layers[self.last_convnet_layer_name + 1 :]

        # Model đầu tiên
        last_conv_layer_model = keras.Model(
            inputs=self.model.inputs, outputs=last_conv_layer.output
        )

        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input

        for layer in classifier_layers:
            x = layer(x)

        # Model thứ hai
        classifier_model = keras.Model(inputs=classifier_input, outputs=x)

        return last_conv_layer_model, classifier_model

    def do_gradient(self, last_conv_layer_model, classifier_model):
        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model(self.images)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        return grads, last_conv_layer_output

    def get_heatmap(self, grads, last_conv_layer_output):
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        last_conv_layer_output = last_conv_layer_output.numpy()[0]

        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(last_conv_layer_output, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

    def convert_1image(self, img, heatmap):

        heatmap = np.uint8(255 * heatmap)

        jet = cm.get_cmap("jet")  # Dùng "jet" để tô màu lại heatmap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)
        superimposed_img = jet_heatmap * 0.4 + img
        superimposed_img = keras.utils.array_to_img(superimposed_img)

        return superimposed_img

    def convert(self):
        last_conv_layer_model, classifier_model = self.create_models()
        grads, last_conv_layer_output = self.do_gradient(
            last_conv_layer_model, classifier_model
        )
        heatmap = self.get_heatmap(grads, last_conv_layer_output)

        list_superimposed_img = [
            self.convert_1image(img, heatmap) for img in self.images
        ]

        return list_superimposed_img


class ImagesToArrayConverter:
    """Chuyển 1 tập ảnh  thành 1 mảng numpy

    Attributes:
        image_paths (list): Tập các đường dẫn đến các file ảnh
        target_size (int): Kích thước sau khi resize

    Hàm -> convert()

    Returns:
        result (np.ndarray):
    """

    def __init__(self, image_paths, target_size):

        self.image_paths = image_paths
        self.target_size = (target_size, target_size)

    def convert_1image(self, img_path):
        img = keras.utils.load_img(
            img_path, target_size=self.target_size
        )  # load ảnh và resize luôn
        array = keras.utils.img_to_array(img)  # Chuyển img sang array
        array = np.expand_dims(
            array, axis=0
        )  # Thêm chiều để tạo thành mảng có 1 phần tử
        return array

    def convert(self):
        return np.vstack(
            [self.convert_1image(img_path) for img_path in self.image_paths]
        )


class ManyConvNetBlocks_XceptionVersion(layers.Layer):
    """Gồm nhiều khối **ConvNetBlocks_XceptionVersion**

    Với list_filters = [32, 64, 128, 256] -> bao gồm các layers:

    - ConvNetBlocks_XceptionVersion(filters = 32)
    - ConvNetBlocks_XceptionVersion(filters = 64)
    - ConvNetBlocks_XceptionVersion(filters = 128)
    - ConvNetBlocks_XceptionVersion(filters = 256)

    Attributes:
        list_filters (list[int]): các filters ứng với từng block **ConvNetBlocks_XceptionVersion**
    """

    def __init__(self, list_filters, **kwargs):
        super().__init__(**kwargs)
        self.list_filters = list_filters

    def build(self, input_shape):
        self.list_ConvNetBlocks = [
            ConvNetBlock_XceptionVersion(filters=filters)
            for filters in self.list_filters
        ]

        super().build(input_shape)

    def call(self, x):
        for convNetBlocks in self.list_ConvNetBlocks:
            x = convNetBlocks(x)

        return x

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update({"list_filters": self.list_filters})
        return config

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class ManyConvNetBlocks_Advanced(layers.Layer):
    """Gồm nhiều block **ConvNetBlocks_Advanced**

    Attributes:
        list_filters (list[int]): list các filters ứng với từng block
    """

    def __init__(self, list_filters, **kwargs):
        super().__init__(**kwargs)
        self.list_filters = list_filters

    def build(self, input_shape):
        self.list_ConvNetBlocks = [
            ConvNetBlock_Advanced(filters=filters) for filters in self.list_filters
        ]

        super().build(input_shape)

    def call(self, x):
        for convNetBlocks in self.list_ConvNetBlocks:
            x = convNetBlocks(x)

        return x

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "list_filters": self.list_filters,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class ManyConvNetBlocks(layers.Layer):
    """Gồm nhiều block **ConvNetBlock**

    Attributes:
        list_filters (list[int]):
        list_num_Conv2D (list[int], optional): Defaults to all 1.
    """

    def __init__(self, list_filters, list_num_Conv2D=None, **kwargs):
        """ """
        # super(ConvNetBlock, self).__init__()
        super().__init__(**kwargs)
        self.list_filters = list_filters
        self.list_num_Conv2D = list_num_Conv2D

    def build(self, input_shape):
        if self.list_num_Conv2D is None:
            self.list_ConvNetBlocks = [
                ConvNetBlock(filters=filters) for filters in self.list_filters
            ]
        elif len(self.list_filters) != len(self.list_num_Conv2D):
            raise ValueError(
                "====== Độ dài của tham số list_filters và list_num_Conv2D phải như nhau ========="
            )
        else:
            self.list_ConvNetBlocks = [
                ConvNetBlock(filters=filters, num_Conv2D=num_Conv2D)
                for filters, num_Conv2D in zip(self.list_filters, self.list_num_Conv2D)
            ]

        super().build(input_shape)

    def call(self, x):
        for convNetBlocks in self.list_ConvNetBlocks:
            x = convNetBlocks(x)

        return x

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "list_filters": self.list_filters,
                "list_num_Conv2D": self.list_num_Conv2D,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomisedModelCheckpoint(keras.callbacks.Callback):
    """Callback để lưu model tốt nhất theo từng epoch

    Attributes:
        filepath (str): đường dẫn đến best model
        monitor (str): chỉ số đánh giá (đánh giá theo **val**), *vd:* val_accuracy, val_loss , ...
        indicator (str): chỉ tiêu

    Examples:
        Với **monitor = val_accuracy và indicator = 0.99**

        Tìm model thỏa val_accuracy > 0.99 và train_accuracy > 0.99 (1) và val_accuracy là lớn nhất trong số đó

        Nếu không thỏa (1) thì lấy theo val_accuracy lớn nhất

    """

    def __init__(self, filepath: str, monitor: str, indicator: float):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.indicator = indicator

    def on_train_begin(self, logs=None):
        self.sign_for_score = (
            1  # Nếu scoring là loss thì lấy âm -> quy về tìm lớn nhất thôi
        )
        if (
            self.monitor.endswith("loss")
            or self.monitor.endswith("mse")
            or self.monitor.endswith("mae")
        ):
            self.indicator = -self.indicator
            self.sign_for_score = -1

        self.per_epoch_val_scores = []
        self.per_epoch_train_scores = []
        self.models = []

    def on_epoch_end(self, epoch, logs=None):
        self.models.append(self.model)
        self.per_epoch_val_scores.append(logs.get(self.monitor) * self.sign_for_score)
        self.per_epoch_train_scores.append(
            logs.get(self.monitor[4:]) * self.sign_for_score
        )

    def on_train_end(self, logs=None):
        # Tìm model tốt nhất
        self.per_epoch_val_scores = np.asarray(self.per_epoch_val_scores)
        self.per_epoch_train_scores = np.asarray(self.per_epoch_train_scores)

        indexs_good_model = np.where(
            (self.per_epoch_val_scores > self.indicator)
            & (self.per_epoch_train_scores > self.indicator)
        )[0]

        index_best_model = None
        if (
            len(indexs_good_model) == 0
        ):  # Nếu ko có model nào đạt chỉ tiêu thì lấy cái tốt nhất
            index_best_model = np.argmax(self.per_epoch_val_scores)
        else:
            val_series = pd.Series(
                self.per_epoch_val_scores[indexs_good_model], index=indexs_good_model
            )
            index_best_model = val_series.idxmax()

        best_model = self.models[index_best_model]

        # Lưu model tốt nhất
        best_model.save(self.filepath)


class ColumnsDeleter(BaseEstimator, TransformerMixin):
    """Xóa cột

    Attributes:
        columns: tên các cột cần xóa
    """

    def __init__(self, columns) -> None:
        super().__init__()
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X = X.drop(columns=self.columns)

        self.cols = X.columns.tolist()
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class ClassifierEvaluator:
    """Đánh giá model cho tập train-val hoặc tập test cho bài toán classification

    Hàm chính:
    - evaluate():

    Lưu ý:
    - KHi đánh giá 1 tập (vd: đánh giá tập test) thì truyền cho train_feature_data, train_target_data, còn val_feature_data và val_target_data **bỏ trống**

    Attributes:
        model (_type_):
        class_names (_type_): Các label
        train_feature_data (_type_):
        train_target_data (_type_):
        val_feature_data (_type_, optional): Defaults to None.
        val_target_data (_type_, optional):  Defaults to None.

    """

    def __init__(
        self,
        model,
        class_names,
        train_feature_data,
        train_target_data,
        val_feature_data=None,
        val_target_data=None,
    ):
        self.model = model
        self.class_names = class_names
        self.train_feature_data = train_feature_data
        self.train_target_data = train_target_data
        self.val_feature_data = val_feature_data
        self.val_target_data = val_target_data

    def evaluate_train_classifier(self):
        train_pred = self.model.predict(self.train_feature_data)
        val_pred = self.model.predict(self.val_feature_data)

        # Accuracy
        train_accuracy = metrics.accuracy_score(self.train_target_data, train_pred)
        val_accuracy = metrics.accuracy_score(self.val_target_data, val_pred)

        # Classification report
        class_names = np.asarray(class_names)
        named_train_target_data = class_names[self.train_target_data]
        named_train_pred = class_names[train_pred]
        named_val_target_data = class_names[self.val_target_data]
        named_val_pred = class_names[val_pred]

        train_classification_report = metrics.classification_report(
            named_train_target_data, named_train_pred
        )
        val_classification_report = metrics.classification_report(
            named_val_target_data, named_val_pred
        )

        # Confusion matrix
        train_confusion_matrix = metrics.confusion_matrix(
            named_train_target_data, named_train_pred, labels=class_names
        )
        np.fill_diagonal(train_confusion_matrix, 0)
        train_confusion_matrix = myfuncs.get_heatmap_for_confusion_matrix_30(
            train_confusion_matrix, class_names
        )

        val_confusion_matrix = metrics.confusion_matrix(
            named_val_target_data, named_val_pred, labels=class_names
        )
        np.fill_diagonal(val_confusion_matrix, 0)
        val_confusion_matrix = myfuncs.get_heatmap_for_confusion_matrix_30(
            val_confusion_matrix, class_names
        )

        model_results_text = f"Train accuracy: {train_accuracy}\n"
        model_results_text += f"Val accuracy: {val_accuracy}\n"
        model_results_text += (
            f"Train classification_report: \n{train_classification_report}\n"
        )
        model_results_text += (
            f"Val classification_report: \n{val_classification_report}"
        )

        return model_results_text, train_confusion_matrix, val_confusion_matrix

    def evaluate_test_classifier(self):
        test_pred = self.model.predict(self.train_feature_data)

        # Accuracy
        test_accuracy = metrics.accuracy_score(self.train_target_data, test_pred)

        # Classification report
        class_names = np.asarray(class_names)
        named_train_target_data = class_names[self.train_target_data]
        named_train_pred = class_names[test_pred]

        test_classification_report = metrics.classification_report(
            named_train_target_data, named_train_pred
        )

        # Confusion matrix
        test_confusion_matrix = metrics.confusion_matrix(
            named_train_target_data, named_train_pred, labels=class_names
        )
        np.fill_diagonal(test_confusion_matrix, 0)
        test_confusion_matrix = myfuncs.get_heatmap_for_confusion_matrix_30(
            test_confusion_matrix, class_names
        )

        model_results_text = f"Test Accuracy: {test_accuracy}\n"
        model_results_text += (
            f"Test Classification_report: \n{test_classification_report}\n"
        )

        return model_results_text, test_confusion_matrix

    def evaluate(self):
        return (
            self.evaluate_train_classifier()
            if self.val_feature_data is not None
            else self.evaluate_test_classifier()
        )


class RegressorEvaluator:
    """Đánh giá model cho tập train-val hoặc tập test cho bài toán regression

    Hàm chính:
    - evaluate():

    Lưu ý:
    - KHi đánh giá 1 tập (vd: đánh giá tập test) thì truyền cho train_feature_data, train_target_data, còn val_feature_data và val_target_data **bỏ trống**

    Attributes:
        model (_type_):
        train_feature_data (_type_):
        train_target_data (_type_):
        val_feature_data (_type_, optional): Defaults to None.
        val_target_data (_type_, optional):  Defaults to None.

    """

    def __init__(
        self,
        model,
        train_feature_data,
        train_target_data,
        val_feature_data=None,
        val_target_data=None,
    ):
        self.model = model
        self.train_feature_data = train_feature_data
        self.train_target_data = train_target_data
        self.val_feature_data = val_feature_data
        self.val_target_data = val_target_data

    def evaluate_train_regressor(self):
        train_pred = self.model.predict(self.train_target_data)
        val_pred = self.model.predict(self.val_target_data)

        # RMSE
        train_rmse = np.sqrt(
            metrics.mean_squared_error(self.train_target_data, train_pred)
        )
        val_rmse = np.sqrt(metrics.mean_squared_error(self.val_target_data, val_pred))

        # MAE
        train_mae = metrics.mean_absolute_error(self.train_target_data, train_pred)
        val_mae = metrics.mean_absolute_error(self.val_target_data, val_pred)

        model_results_text = f"Train RMSE: {train_rmse}\n"
        model_results_text += f"Val RMSE: {val_rmse}\n"
        model_results_text += f"Train MAE: {train_mae}\n"
        model_results_text += f"Val MAE: {val_mae}\n"

        return model_results_text

    def evaluate_test_regressor(self):
        test_pred = self.model.predict(self.train_target_data)

        # RMSE
        test_rmse = np.sqrt(
            metrics.mean_squared_error(self.train_target_data, test_pred)
        )

        # MAE
        test_mae = metrics.mean_absolute_error(self.train_target_data, test_pred)

        model_results_text = f"Test RMSE: {test_rmse}\n"
        model_results_text = f"Test MAE: {test_mae}\n"

        return model_results_text


class BestModelSearcher:
    """Searcher đi tìm model tốt nhất và train, val scoring tương ứng

    Hàm chính:
        - next()

    Examples:
        Với **scoring = accuracy và target_score = 0.99**

        Tìm model thỏa val_accuracy > 0.99 và train_accuracy > 0.99 (1) và val_accuracy là lớn nhất trong số đó

        Nếu không thỏa (1) thì lấy theo val_accuracy lớn nhất

    Attributes:
        models (_type_): Model tốt nhất đang ở trong này
        train_scorings (_type_):
        val_scorings (_type_):
        target_score (_type_): Chỉ tiêu đề ra
        scoring (_type_): Chỉ số đánh giá


    """

    def __init__(self, models, train_scorings, val_scorings, target_score, scoring):
        self.models = models
        self.train_scorings = train_scorings
        self.val_scorings = val_scorings
        self.target_score = target_score
        self.scoring = scoring

    def find_train_val_scorings_to_find_the_best(self):
        sign_for_score = 1  # Nếu scoring cần min thì lấy âm -> quy về tìm lớn nhất thôi
        if self.scoring in myfuncs.SCORINGS_PREFER_MININUM:
            self.target_score = -self.target_score
            sign_for_score = -1

        self.train_scorings_to_find_the_best = np.asarray(
            [item * sign_for_score for item in self.train_scorings]
        )
        self.val_scorings_to_find_the_best = np.asarray(
            [item * sign_for_score for item in self.val_scorings]
        )

    def next(self):
        self.find_train_val_scorings_to_find_the_best()

        indexs_good_model = np.where(
            (self.val_scorings_to_find_the_best > self.target_score)
            & (self.train_scorings_to_find_the_best > self.target_score)
        )[0]

        index_best_model = None
        if (
            len(indexs_good_model) == 0
        ):  # Nếu ko có model nào đạt chỉ tiêu thì lấy cái tốt nhất
            index_best_model = np.argmax(self.val_scorings_to_find_the_best)
        else:
            val_series = pd.Series(
                self.val_scorings_to_find_the_best[indexs_good_model],
                index=indexs_good_model,
            )
            index_best_model = val_series.idxmax()

        best_model = self.models[index_best_model]
        train_scoring = self.train_scorings[index_best_model]
        val_scoring = self.val_scorings[index_best_model]

        return best_model, index_best_model, train_scoring, val_scoring


class CustomStackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, final_estimator):
        self.estimators = estimators
        self.final_estimator = final_estimator

    def fit(self, X, y):
        for estimator in self.estimators:
            estimator.fit(X, y)

        new_feature = self.get_new_feature_through_estimators(X)

        self.final_estimator.fit(new_feature, y)

        return self

    def predict(self, X):
        new_feature = self.get_new_feature_through_estimators(X)

        return self.final_estimator.predict(new_feature)

    def predict_proba(self, X):

        new_feature = self.get_new_feature_through_estimators(X)

        return self.final_estimator.predict_proba(new_feature)

    def get_new_feature_through_estimators(self, X):
        """Get new feature thông qua các estimators

        VD: nếu có 3 estimators và có 4 label cần phân loại thì kích thước của kết quả là (N, 3 * 4) = (N, 12)

        với N: số sample
        """
        list_predict_proba = [
            estimator.predict_proba(X) for estimator in self.estimators
        ]
        new_feature = np.hstack(*list_predict_proba)

        return new_feature
