import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from title_dataset import TitleDataset
from class_idxs import ind2label, ind2crop

generator = tf.keras.models.load_model("data/model_v3_001041.h5")

npzfile = np.load("data/train_data_real.npz")
dataset = TitleDataset(npzfile, {}, is_train=False)


def read_data(index):
    real, labels = dataset[index]
    np.nan_to_num(real, copy=False)
    real = real + 0.25

    return real, labels


def generate_image(labels):
    one_hot = np.zeros((1, labels.shape[0], labels.shape[1], 16), dtype=np.uint8)

    for i in range(16):
        one_hot[0, :, :, i][i == labels] = 1
    fake = 0.25 + generator.predict(one_hot)[0, :, :, :]

    np.nan_to_num(fake, copy=False)

    return fake


def get_stat_from_set(dataset):
    all_pix_valid_titles = {}
    for i in ind2crop.keys():
        all_pix_valid_titles[i] = 0

    for _, labels in dataset:

        crops_pixels = 0
        for i in ind2crop.keys():
            crops_pixels += np.sum(labels == i)
        if not (crops_pixels > 0.3 * labels.size):
            continue

        for i in ind2crop.keys():
            all_pix_valid_titles[i] += np.sum(labels == i)

    return all_pix_valid_titles


def get_new_data(all_pix_valid_titles, dataset):
    labes_by_rise = sorted(ind2crop.keys(), key=lambda x: all_pix_valid_titles[x])
    labes_by_fall = reversed(labes_by_rise)

    old2new_label_map = {}
    for i, j in zip(labes_by_rise, labes_by_fall):
        old2new_label_map[i] = j

    all_pix_valid_titles2 = all_pix_valid_titles.copy()

    all_fake_data = []
    all_fake_labels = []
    for _, labels in dataset:

        crops_pixels = 0
        for i in ind2crop.keys():
            crops_pixels += np.sum(labels == i)
        if not (crops_pixels > 0.3 * labels.size):
            continue

        labels = np.vectorize(lambda x: old2new_label_map.get(x, x))(labels)
        fake = generate_image(labels)
        all_fake_data.append(fake)
        all_fake_labels.append(labels)

        for i in ind2crop.keys():
            all_pix_valid_titles2[i] += np.sum(labels == i)

    all_fake_data = np.array(all_fake_data)
    all_fake_labels = np.array(all_fake_labels).astype(np.uint8)

    return all_fake_data, all_fake_labels, all_pix_valid_titles2


all_pix_valid_titles = get_stat_from_set(dataset)
fake_data1, fake_labels1, all_pix_valid_titles2 = get_new_data(all_pix_valid_titles, dataset)
fake_data2, fake_labels2, all_pix_valid_titles3 = get_new_data(all_pix_valid_titles2, dataset)

plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
plt.pie([all_pix_valid_titles[i] for i in ind2crop.keys()], labels=[ind2label[i] for i in ind2crop.keys()])
plt.title("Real data distribution (crops)")
plt.subplot(1, 3, 2)
plt.pie([all_pix_valid_titles2[i] for i in ind2crop.keys()], labels=[ind2label[i] for i in ind2crop.keys()])
plt.title("Real+Fake data distribution (crops)")
plt.subplot(1, 3, 3)
plt.pie([all_pix_valid_titles3[i] for i in ind2crop.keys()], labels=[ind2label[i] for i in ind2crop.keys()])
plt.title("Real+Fake+Fake data distribution (crops)")
plt.savefig("data_distribution.pdf")
plt.show()

all_data = np.concatenate([fake_data1, fake_data2])
all_labels = np.concatenate([fake_labels2, fake_labels2])

to_save_dict = {}
for i in range(all_labels.shape[0]):
    to_save_dict["train_data_fake_" + str(i)] = all_data[i]
    to_save_dict["train_label_" + str(i)] = all_labels[i]

np.savez("data/train_data_gan.npz", **to_save_dict)
