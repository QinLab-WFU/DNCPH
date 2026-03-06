import configparser
import os
import platform

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MyDataset(Dataset):
    """
    Common dataset for DeepHashing.

    Args
        dataset(str): Dataset name.
        img_dir(str): Directory of image files.
        txt_dir(str): Directory of txt file containing image file names & image labels.
        transform(callable, optional): Transform images.
    """

    def __init__(self, root, dataset, usage, transform=None):
        assert dataset in ["cifar", "flickr", "coco", "nuswide"]
        self.name = dataset

        # 根据数据集名称映射到实际文件夹路径
        if dataset == "cifar":
            data_path = os.path.join(root, "CIFAR-10")
            self.num_classes = 10
        elif dataset == "flickr":
            data_path = os.path.join(root, "MIRFLICKR-25K")
            self.num_classes = 38
        elif dataset == "coco":
            data_path = os.path.join(root, "MS-COCO")
            self.num_classes = 80
        elif dataset == "nuswide":
            data_path = os.path.join(root, "NUS-WIDE-TC21")
            self.num_classes = 21
        else:
            raise NotImplementedError(f"unknown dataset: {dataset}")



        assert usage in ["train", "test", "database"]

        self.usage = usage

        if not os.path.exists(root):
            print(f"root not exists: {root}")
            root = os.path.dirname(__file__) + "/_datasets"
            print(f"root will use: {root}")

        # xxx_dir = os.path.join(root, f"{dataset}")


        # img_dir = f"{data_path}/images"
        img_dir = f"{data_path}"
        # img_loc = os.path.join(img_dir, "images_location.txt")
        ini_loc = os.path.join(img_dir, "images_location.ini")
        if os.path.exists(ini_loc):
            # self.img_dir = open(img_loc, "r").readline()
            config = configparser.ConfigParser()
            config.read(ini_loc)
            self.img_dir = config["DEFAULT"][platform.system()]
        else:
            self.img_dir = img_dir
        self.transform = build_default_trans(usage) if transform is None else transform

        # Read files
        self.data = [
            (x.split()[0], np.array(x.split()[1:], dtype=np.float32))
            for x in open(os.path.join(data_path, f"{usage}.txt"), "r").readlines()
        ]

    def __getitem__(self, index):
        file_name, label = self.data[index]
        with open(os.path.join(self.img_dir, file_name), "rb") as fp:
            img = Image.open(fp).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        return torch.from_numpy(np.vstack([x[1] for x in self.data]))
    def get_all_labels(self):
        # 提取所有样本的标签（每个样本的标签是 self.data[i][1]），堆叠成二维numpy数组后转tensor
        all_labels = np.vstack([x[1] for x in self.data])
        return torch.from_numpy(all_labels)


def get_class_num(dataset):
    r = {"cifar": 10, "flickr": 38, "nuswide": 21, "coco": 80}[dataset]
    return r


def get_topk(dataset):
    r = {"cifar": -1, "flickr": -1, "nuswide": 5000, "coco": -1}[dataset]
    return r
# def get_topk(dataset):
#     r = {"cifar": 15, "flickr": 15, "nuswide": 5000, "coco": 15}[dataset]
#     return r

def build_default_trans(usage, resize_size=256, crop_size=224):
    if usage == "train":
        # step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
        step = [transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip()]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose(
        [transforms.Resize(resize_size)]
        + step
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_loader(root, dataset, usage, transform,  **kwargs):
    dset = MyDataset(root, dataset, usage, transform)
    print(f"{usage} set length: {len(dset)}")
    if "shuffle" in kwargs:
        loader = DataLoader(dset, **kwargs)
    else:
        loader = DataLoader(dset, shuffle=usage == "train", **kwargs)
    return loader


def build_loaders(root, dataset, trans_train=None, trans_test=None, **kwargs):
    loaders = []
    for usage in ["train", "test", "database"]:
        loaders.append(
            build_loader(
                root,
                dataset,
                usage,
                trans_train if usage == "train" else trans_test,
                **kwargs,
            )
        )

    return loaders

def build_trans(usage, resize_size=256, crop_size=224):
    if usage == "train":
        steps = [T.RandomCrop(crop_size), T.RandomHorizontalFlip()]
    else:
        steps = [T.CenterCrop(crop_size)]
    return T.Compose(
        [T.Resize(resize_size)]
        + steps
        + [
            T.ToTensor(),
            # T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

def build_loaders1(name, root, **kwargs):
    train_trans = build_trans("train")
    other_trans = build_trans("test")

    # 构建三个数据集加载器
    train_loader = build_loader(root, name, "train", train_trans, **kwargs)
    query_loader = build_loader(root, name, "test", other_trans, **kwargs)
    dbase_loader = build_loader(root, name, "database", other_trans, **kwargs)

    # data = init_dataset(name, root)
    #
    # train_loader = DataLoader(ImageDataset(data.train, train_trans), shuffle=True, drop_last=True, **kwargs)
    # # generator=torch.Generator(): to keep torch.get_rng_state() unchanged!
    # # https://discuss.pytorch.org/t/does-a-dataloader-change-random-state-even-when-shuffle-argument-is-false/92569/4
    # query_loader = DataLoader(ImageDataset(data.query, other_trans), generator=torch.Generator(), **kwargs)
    # dbase_loader = DataLoader(ImageDataset(data.dbase, other_trans), generator=torch.Generator(), **kwargs)

    return train_loader, query_loader, dbase_loader


class ImageDataset(Dataset):
    """Image Dataset"""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        img, lab = self.data[0][idx], self.data[1][idx]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        else:
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, lab, idx

    def get_all_labels(self):
        return torch.from_numpy(self.data[1])

if __name__ == "__main__":
    _, query_loader, _ = build_loaders("/home/virtue/singlemodal/Dataset", "flickr", batch_size=64, num_workers=3)
    print("topk", get_topk("flickr"))
    print("num_classes", get_class_num("flickr"))
    for x in query_loader:
        print(x[0], x[1])
        break
