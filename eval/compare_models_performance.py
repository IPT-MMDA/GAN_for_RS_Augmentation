import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import os
from osgeo import gdal
from sklearn.metrics import classification_report, jaccard_score

from unet_model import get_unet
from class_idxs import label_map, ind2label


def get_acc(model, images_path):
    correct = 0
    total = 0
    per_label_correct = {}
    per_label_total = {}
    for i in range(16):
        per_label_correct[i] = 0
        per_label_total[i] = 0

    all_labels_gt = []
    all_labels_pred = []
    for w, input_file in enumerate(images_path):
        raster = gdal.Open(input_file)
        if w % 100 == 0:
            print("[{}/{}]".format(w, len(images_path)))
        arr1 = raster.GetRasterBand(1).ReadAsArray()
        arr2 = raster.GetRasterBand(2).ReadAsArray()
        arr3 = raster.GetRasterBand(3).ReadAsArray()
        arr4 = raster.GetRasterBand(4).ReadAsArray()
        label = raster.GetRasterBand(5).ReadAsArray().astype(np.uint8)
        label = np.vectorize(label_map.get)(label)

        img = np.array([arr1, arr2, arr3, arr4])
        img = img.transpose(1, 2, 0)
        np.nan_to_num(img, copy=False)

        img = (img - 0.25)

        img = np.expand_dims(img, 0)
        predicted_label = np.argmax(model.predict(img, verbose=0)[0], axis=-1).astype(np.uint8)

        correct += np.sum(predicted_label == label)
        total += label.size

        all_labels_gt.append(label)
        all_labels_pred.append(predicted_label)

        for i in range(16):
            per_label_correct[i] += np.sum(predicted_label[label == i] == i)
            per_label_total[i] += np.sum(label == i)

    accuracy = correct / total

    per_class_accuracy = [per_label_correct[i] / per_label_total[i] for i in range(16)]

    all_labels_gt = np.array(all_labels_gt).reshape(-1)
    all_labels_pred = np.array(all_labels_pred).reshape(-1)
    return accuracy, per_class_accuracy, all_labels_gt, all_labels_pred


def eval_model(model_path):
    print("===> Evaluate model:", model_path)
    model.load_weights(model_path)
    accuracy_fake, per_class_accuracy_fake, f_gt, f_pred = get_acc(model, images_path)
    print(classification_report(f_gt, f_pred, digits=3))  # , labels=[ind2label[i] for i in ind2label.keys()] ))

    print("\njaccard_score:", jaccard_score(f_gt, f_pred, average="weighted"))
    print("\njaccard_score per class:")
    t = jaccard_score(f_gt, f_pred, average=None)
    for name, value in zip([ind2label[i] for i in ind2label.keys()], np.round(100 * t, 2)):
        print(name, value)

    del f_gt
    del f_pred

    print()
    per_class_accuracy_fake = np.array(per_class_accuracy_fake)
    print("Accuracy (real+fake data): {}%".format(np.round(100 * accuracy_fake, 3)))

    print("\naccuracy per class:")
    for i, value in enumerate(per_class_accuracy_fake):
        print(ind2label[i], np.round(100 * value, 2))


images_path = [os.path.join("cropped_256x256/test", x)
               for x in os.listdir("cropped_256x256/test")
               if x.endswith(".tif")]

im_width = 256
im_height = 256
input_img = tf.keras.layers.Input((im_height, im_width, 4), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                   metrics=["accuracy"])

eval_model("data/model_segmentation_real.h5")
eval_model("data/model_segmentation_gan.h5")
eval_model("data/model_segmentation_replace.h5")
eval_model("data/model_segmentation_stat.h5")
