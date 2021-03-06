DOWNLOADED_MODEL_NAME = "tf_model.tar.gz"
VERSION = "1"

# Keras constants
COLOR_CHANNELS = (3,)
TRAIN_DATA_GENERATOR_KWARGS = {"rescale": 1.0 / 255, "validation_split": 0.20, "rotation_range":10, "brightness_range":[0.8,1.2]}
DATA_GENERATOR_KWARGS = {"rescale": 1.0 / 255, "validation_split": 0.20}
DROPOUT_RATE = 0.2
FROM_LOGITS = True
IMAGE_SIZE = (224, 224)
INTERPOLATION = "bilinear"
LABEL_SMOOTHING = 0.1
REGULARIZERS_L2 = 0.0001
VERBOSE_ONE_LINE_PER_EPOCH = 2
CLASS_LABEL_TO_PREDICTION_INDEX_JSON = "class_label_to_prediction_index.json"
