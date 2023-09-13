import numpy as np
import random


class TitleDataset:
    def __init__(self, real, fake, is_train=False):
        self.real = real
        self.fake = fake
        self.real_keys = [x for x in real.keys() if "_data_" in x]
        self.fake_keys = [x for x in fake.keys() if "_data_" in x]
        self.united_keys = self.real_keys + self.fake_keys
        self.is_train = is_train

    def __len__(self):
        return len(self.real_keys) + len(self.fake_keys)

    def __getitem__(self, idx):
        key = self.united_keys[idx]
        if key in self.real_keys:
            data = self.real[key]
            key = key.replace("_data_real_", "_label_")
            key = key.replace("_data_fake_", "_label_")
            label = self.real[key]
        else:
            data = self.fake[key]
            key = key.replace("_data_real_", "_label_")
            key = key.replace("_data_fake_", "_label_")
            label = self.fake[key]

        if self.is_train:
            if np.random.random() > 0.5:
                data = data[:, ::-1]
                label = label[:, ::-1]
            if np.random.random() > 0.5:
                data = data[::-1]
                label = label[::-1]
            if np.random.random() > 0.5:
                data = np.rot90(data)
                label = np.rot90(label)

        np.nan_to_num(data, copy=False)
        data = (data - 0.25)

        return data, label

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()

    # shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        self.united_keys = random.sample(self.united_keys, k=len(self.united_keys))
