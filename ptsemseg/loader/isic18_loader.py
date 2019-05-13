import os
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import collections

from torch.utils import data

from ptsemseg.utils import recursive_glob


class ISIC18Loader(data.Dataset):
    def __init__(self,
                 root,
                 split="training",
                 is_transform=True,
                 img_size=(1024, 768),
                 augmentations=None,
                 img_norm=True,
                 test_mode=False
                 ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = collections.defaultdict(list)

        if not self.test_mode:
            for folder in ["training", "training_labels", "validation" ]:
                file_list = recursive_glob(
                    rootdir=self.root + "/" + folder + "/", suffix=".png" if folder == "training_labels" else ".jpg"
                )
                self.files[folder] = file_list

    def __len__(self):
        return min(len(self.files[self.split]), len(self.files["training_labels"]))

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        if self.split == "training":
            lbl_name = self.files["training_labels"][index]
            print img_name
        # "ISIC" + str(index).zfill(6) + ".jpg"

        img = m.imread(img_name, mode="RGB")
        img = np.array(img, dtype=np.uint8)

        if self.split == "training":
            lbl = m.imread(lbl_name, mode="L")
            lbl = np.array(lbl, dtype=np.uint8)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl if self.split == "training" else None

    def encode_segmap(self, mask):
        # TODO
        return np.where(mask >= 127, 255, 0)

    def decode_segmap(self, temp, plot=False):
        # TODO
        return temp

    def transform(self, img, lbl):
        # TODO colors dont seem right
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]))
        lbl = self.encode_segmap(lbl)
        return torch.from_numpy(img), torch.from_numpy(lbl)

    def augmentations(self, img, lbl):
        """TODO

        :param img:
        :param lbl:
        :return:
        """
        pass


if __name__ == "__main__":
    local_path = "/home/bijan/Workspace/Python/pytorch-semseg/data/ISIC"
    dst = ISICLoader(local_path, is_transform=True)
    train_loader = data.DataLoader(dst, batch_size=4)
    for i, data_samples in enumerate(train_loader):
        imgs, labels = data_samples
        print i, imgs.size()
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            for j in range(4):
                plt.imshow(dst.decode_segmap(labels.numpy()[j]))
                plt.show()
