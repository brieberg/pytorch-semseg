import random
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import collections

from torch.utils import data

from ptsemseg.utils import recursive_glob


class ISIC18Loader(data.Dataset):
    def __len__(self):
        check_labels = self.split == "training" or self.split == "training"
        return min(len(self.files[self.split]),
                   len(self.files[self.split + "_labels"])) \
            if check_labels else len(self.files[self.split])

    def __init__(self,
                 root,
                 split="training",
                 is_transform=True,
                 img_size=(1024, 768),
                 augmentations=None,
                 img_norm=True,
                 test_mode=False
                 ):
        self.sub_folder = {"training": "training",
                           "training_labels": "training_labels",
                           "validation": "validation",
                           "final_test": "test"}
        test_percentage = 0.1

        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = collections.defaultdict(list)
        self.n_classes = 2

        random.seed(123)
        test_indices = random.sample(range(0, 2594), int(2594*test_percentage))

        if not self.test_mode:
            for folder in [self.sub_folder["training"],
                           self.sub_folder["training_labels"],
                           self.sub_folder["validation"],
                           self.sub_folder["final_test"]]:
                file_list = recursive_glob(
                    rootdir=self.root + "/" + folder + "/",
                    suffix=".png" if folder == self.sub_folder["training_labels"] else ".jpg"
                )
                if folder == self.sub_folder["training"]:
                    self.files["test"] = [sorted(file_list)[k] for k in test_indices]
                    print(self.files["test"])
                    self.files["training"] = sorted(list(set(file_list) -
                                                         set(self.files["test_labels"]))
                                                    )
                elif folder == self.sub_folder["training_labels"]:
                    self.files["test_labels"] = [sorted(file_list)[k] for k in test_indices]
                    self.files["training_labels"] = sorted(list(set(file_list) -
                                                                set(self.files["test_labels"]))
                                                           )
                    print(self.files["test_labels"])
                elif folder == self.sub_folder["validation"]:
                    self.files["validation"] = sorted(file_list)
                else:
                    self.files["final_test"] = sorted(file_list)

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        if self.split == "training" or self.split == "test":
            lbl_name = self.files[self.split + "_labels"][index]

        # load and convert RGB to BGR
        img = m.imread(img_name, mode="RGB")  # [..., [2, 0, 1]]
        lbl = None

        if self.split == "training" or self.split == "test":
            lbl = m.imread(lbl_name, mode="L")
            lbl = np.array(lbl, dtype=np.uint8)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        if lbl is not None:
            lbl[lbl < 1] = 0  # Nothing
            lbl[lbl >= 1] = 1  # Something

        return img, lbl if self.split == "training" or self.split == "test" else None

    @staticmethod
    def decode_segmap(lbl):
        lbl[lbl < 1] = 0     # Nothing
        lbl[lbl >= 1] = 255  # Something
        return lbl

    def transform(self, image, label):
        if self.img_size == ("same", "same"):
            pass

        image = m.imresize(image, (self.img_size[0], self.img_size[1]))

        # NxHxWxC -> NxCxHxW
        image = image.transpose(2, 0, 1)

        if label is not None:
            label = m.imresize(label, (self.img_size[0], self.img_size[1]))

        return torch.Tensor(image), torch.Tensor(label) if label is not None else None

    def augmentations(self, img, lbl):
        """TODO

        :param img:
        :param lbl:
        :return:
        """
        return img, lbl


if __name__ == "__main__":
    local_path = "/home/bijan/Workspace/Python/pytorch-semseg/data/ISIC18"
    dst = ISIC18Loader(local_path, img_size=(572, 572), split="test", is_transform=True)
    print(dst.__len__())
    batch_size = 4
    train_loader = data.DataLoader(dst,
                                   batch_size=batch_size,
                                   num_workers=4,
                                   shuffle=True)

    for i, data_samples in enumerate(train_loader):
        imgs, labels = data_samples
        print(imgs.dim())
        if i in range(0, 4):
            img = torchvision.utils.make_grid(imgs).numpy()
            img = (np.transpose(img, (1, 2, 0)))  # [..., ::-1]
            plt.imshow(img)
            plt.show()

            fig, ax = plt.subplots(1, batch_size,
                                   figsize=(10,4),
                                   sharey=True,
                                   dpi=120)

            for j in range(batch_size):
                ax[j].imshow(labels.numpy()[j])
            plt.show()
        else:
            break
