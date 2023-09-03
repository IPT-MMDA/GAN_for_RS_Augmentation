import os
from osgeo import gdal

import numpy as np
from tqdm import tqdm_notebook as tqdm
import random

from class_idxs import label_map

tif_list = [fname for fname in os.listdir("cropped_256x256") if fname.endswith(".tif")]
train_set = random.sample(tif_list, k=len(tif_list) // 2)

for fname in tqdm(os.listdir("cropped_256x256")):
    if not fname.endswith(".tif"):
        continue

    if fname in train_set:
        os.rename(os.path.join("cropped_256x256", fname),
                  os.path.join("cropped_256x256/train", fname))
    else:
        os.rename(os.path.join("cropped_256x256", fname),
                  os.path.join("cropped_256x256/test", fname))


def make_npz_file(target_dir, out_file):
    all_data = []
    all_labels = []
    for fname in os.listdir(target_dir):
        if not fname.endswith(".tif"):
            continue
        raster = gdal.Open(os.path.join(target_dir, fname))
        arr1 = raster.GetRasterBand(1).ReadAsArray()
        arr2 = raster.GetRasterBand(2).ReadAsArray()
        arr3 = raster.GetRasterBand(3).ReadAsArray()
        arr4 = raster.GetRasterBand(4).ReadAsArray()
        arr5 = raster.GetRasterBand(5).ReadAsArray()

        img = np.array([arr1, arr2, arr3, arr4])
        img = img.transpose(1, 2, 0)
        np.nan_to_num(img, copy=False)
        all_data.append(img)

        np.nan_to_num(arr5, copy=False)
        all_labels.append(arr5)
    all_labels = np.array(all_labels)
    all_data = np.array(all_data)

    all_labels = all_labels.astype(np.uint8)
    all_labels = np.vectorize(label_map.get)(all_labels)

    to_save_dict = {}
    for i in range(all_labels.shape[0]):
        to_save_dict["test_data_real_" + str(i)] = all_data[i]
        to_save_dict["test_label_" + str(i)] = all_labels[i]

    np.savez(out_file, **to_save_dict)


make_npz_file("cropped_256x256/train", "data/train_data_real.npz")
make_npz_file("cropped_256x256/test", "data/test_data.npz")
