"""transfer_learning_with_hub.ipynb
https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub
https://www.tensorflow.org/hub/tutorials/tf2_image_retraining
https://www.tensorflow.org/tfx/tutorials/serving/rest_simple
For improving data loading:
https://www.tensorflow.org/tutorials/load_data/images
https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#training-with-pipe
-mode-using-pipemodedataset
"""

import argparse
import json
import logging
import os
import sys
import tarfile

import boto3
import tensorflow as tf
from constants import constants
import time
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
root.addHandler(handler)

import smdistributed.dataparallel.tensorflow as dist

dist.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    # SMDataParallel: Pin GPUs to a single SMDataParallel process [use SMDataParallel local_rank() API]
    tf.config.experimental.set_visible_devices(gpus[dist.local_rank()], 'GPU')


def download_from_s3(bucket, key, local_rel_dir, model_name):
    local_model_path = os.path.join(os.path.dirname(__file__), local_rel_dir, model_name)
    client = boto3.client("s3")
    client.download_file(bucket, key, local_model_path)


def prepare_model(model_artifact_bucket, model_artifact_key, num_labels):
    if dist.local_rank() == 0:
        download_from_s3(
            bucket=model_artifact_bucket,
            key=model_artifact_key,
            local_rel_dir=".",
            model_name=constants.DOWNLOADED_MODEL_NAME,
        )
        with tarfile.open(constants.DOWNLOADED_MODEL_NAME) as saved_model_tar:
            saved_model_tar.extractall(constants.DOWNLOADED_MODEL_NAME.replace(".tar.gz", ""))
    else: 
        while True: 
            if os.path.exists(constants.DOWNLOADED_MODEL_NAME.replace(".tar.gz", "/" + constants.VERSION)): 
                break
            else: 
                time.sleep(10)
        
    feature_extractor_layer = tf.keras.models.load_model(
        constants.DOWNLOADED_MODEL_NAME.replace(".tar.gz", "/" + constants.VERSION)
    )
    feature_extractor_layer.trainable = False
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=constants.IMAGE_SIZE + constants.COLOR_CHANNELS),
            feature_extractor_layer,
            tf.keras.layers.Dropout(rate=constants.DROPOUT_RATE),
            tf.keras.layers.Dense(
                num_labels,
                kernel_regularizer=tf.keras.regularizers.l2(constants.REGULARIZERS_L2),
            ),
        ]
    )
    model.build((None,) + constants.IMAGE_SIZE + constants.COLOR_CHANNELS)
    return model


def prepare_data(data_dir, batch_size):
    dataflow_kwargs = dict(
        target_size=constants.IMAGE_SIZE, batch_size=batch_size, interpolation=constants.INTERPOLATION, 
    )
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**constants.DATA_GENERATOR_KWARGS)
    valid_generator = valid_datagen.flow_from_directory(data_dir, subset="validation", shuffle=False, **dataflow_kwargs)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**constants.TRAIN_DATA_GENERATOR_KWARGS)
    train_generator = train_datagen.flow_from_directory(data_dir, subset="training", shuffle=True, **dataflow_kwargs)

    return train_generator, valid_generator


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the
    # default bucket.
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    parser.add_argument("--model-artifact-bucket", type=str)
    parser.add_argument("--model-artifact-key", type=str)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--adam-learning-rate", type=float, default=0.001)

    return parser.parse_known_args()

@tf.function
def training_step(model, loss, opt, images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = model(images, training=True)
        loss_value = loss(labels, probs)

    # SMDataParallel: Wrap tf.GradientTape with SMDataParallel's DistributedGradientTape
    tape = dist.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    if first_batch:
       # SMDataParallel: Broadcast model and optimizer variables
        dist.broadcast_variables(model.variables, root_rank=0)
        dist.broadcast_variables(opt.variables(), root_rank=0)
    
    # SMDataParallel: all_reduce call
    loss_value = dist.oob_allreduce(loss_value) # Average the loss across workers
    return loss_value



def run_with_args(args):
    train_generator, valid_generator = prepare_data(data_dir=args.train, batch_size=args.batch_size)
    logging.info(f"prediction class indices mapping to input training data labels: {train_generator.class_indices}")

    model = prepare_model(
        model_artifact_bucket=args.model_artifact_bucket,
        model_artifact_key=args.model_artifact_key,
        num_labels=train_generator.num_classes,
    )
    model.summary()
    optimizer=tf.keras.optimizers.Adam(args.adam_learning_rate*dist.size())
    loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=constants.FROM_LOGITS, label_smoothing=constants.LABEL_SMOOTHING
    )
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )
    

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    steps_per_evaluate = valid_generator.samples // train_generator.batch_size
    print('total', train_generator.samples )
    print('batch', train_generator.batch_size)
    batch = 0 

    while batch < steps_per_epoch*args.epochs: 
        info = train_generator.next()
        images = info[0]
        print(images.shape)
        labels = info[1]
        print(labels.shape)
        batch +=1 
        loss_value = training_step(model, loss, optimizer, images, labels, batch == 0)

        if batch % 50 == 0 :
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))
            Y_pred = model.predict_generator(valid_generator, steps_per_evaluate+1)
            y_pred = np.argmax(Y_pred, axis=1)
            print('Confusion Matrix')
            print(confusion_matrix(valid_generator.classes, y_pred))
            target_names=[ 'False', 'True']
            print(classification_report(valid_generator.classes, y_pred, target_names=target_names))
            

    

    if  dist.rank() == 0:
        model_uncompiled = prepare_model(
            model_artifact_bucket=args.model_artifact_bucket,
            model_artifact_key=args.model_artifact_key,
            num_labels=train_generator.num_classes,
        )
        model_uncompiled.set_weights(model.get_weights())
        export_path = os.path.join(args.model_dir, constants.VERSION)
        tf.keras.models.save_model(
            model_uncompiled,
            export_path,
            overwrite=True,
            include_optimizer=False,
            save_format=None,
            signatures=None,
            options=None,
        )
        # saving class_label_to_prediction_index in model_dir
        with open(os.path.join(args.model_dir, constants.CLASS_LABEL_TO_PREDICTION_INDEX_JSON), "w") as jsonFile:
            json.dump(train_generator.class_indices, jsonFile)


if __name__ == "__main__":
    args, unknown = _parse_args()
    run_with_args(args)
