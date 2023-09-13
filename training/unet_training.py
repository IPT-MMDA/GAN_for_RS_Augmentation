#!/usr/bin/env python
import numpy as np

import tensorflow as tf
import argparse
import os

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from title_dataset import TitleDataset
from unet_model import get_unet


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', type=str, required=True,
                        help='location for writing checkpoints and exporting models')
    parser.add_argument('--type', type=str, required=True,
                        choices=['real', 'gan', 'replace', 'stat'],
                        help='training type')
    args, _ = parser.parse_known_args()
    return args


def main(args):
    work_dir = args.job_dir
    training_type = args.type
    model_out_file = os.path.join(work_dir, "model_segmentation_{}.h5".format(training_type))

    im_width = 256
    im_height = 256

    print("===> Loading dataset...")

    npzfile_train_real = np.load(os.path.join(work_dir, "train_data_real.npz"))

    if training_type != "real":
        fake_path = os.path.join(work_dir, "train_data_{}.npz".format(training_type))
        npzfile_train_fake = np.load(fake_path)
    else:
        npzfile_train_fake = {}

    train_datagenerator = TitleDataset(npzfile_train_real, npzfile_train_fake, is_train=True)
    ds_train = tf.data.Dataset.from_generator(train_datagenerator,
                                              output_signature=(
                                                  tf.TensorSpec(shape=(256, 256, 4), dtype=tf.float32),
                                                  tf.TensorSpec(shape=(256, 256), dtype=tf.int32),
                                              )
                                              )
    ds_train = ds_train.batch(32)

    npzfile_test = np.load(os.path.join(work_dir, "test_data.npz"))
    test_datagenerator = TitleDataset(npzfile_test, {}, is_train=False)
    ds_test = tf.data.Dataset.from_generator(test_datagenerator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(256, 256, 4), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(256, 256), dtype=tf.int32),
                                             )
                                             )
    ds_test = ds_test.batch(32)

    print("===> Building model...")

    input_img = Input((im_height, im_width, 4), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    model.compile(optimizer=Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint(model_out_file, verbose=1, save_best_only=True,
                        save_weights_only=True)
    ]

    print("===> Starting training...")

    results = model.fit(ds_train, epochs=60, callbacks=callbacks,
                        validation_data=ds_test)

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",
             label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig(os.path.join(work_dir, "plot_segmentation_{}.pdf".format(training_type)))

    print("===> Evaluating model...")

    model.load_weights(model_out_file)
    model.evaluate(ds_test, verbose=1)


main(get_args())
