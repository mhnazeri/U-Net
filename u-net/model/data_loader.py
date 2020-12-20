"""Creating a dataset for carla image segmentation"""
from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CarlaSeg(Dataset):
    """Carla Image Segmentation Class"""

    def __init__(
        self,
        root: str = "../data",
        n_classes: int = 23,
        img_size: Union[int, Tuple[int, int]] = (256, 256),
        mode: str = "train",
    ):
        # read directories
        self.n_classes = n_classes
        if mode.lower() == "train":
            dirs = [
                "dataA", "dataB", "dataC", "dataD"
            ]
        elif mode.lower() == "val":
            dirs = ["dataE"]
        else:
            raise ValueError(
                "Unknown parameter for mode, it should be 'train' or 'val'"
            )

        root = Path(root)
        img_dirs = [Path(root / x / "CameraRGB") for x in dirs]
        seg_dirs = [Path(root / x / "CameraSeg") for x in dirs]
        self.img_address = sorted([x for d in img_dirs for x in d.iterdir()])
        self.seg_address = sorted([x for d in seg_dirs for x in d.iterdir()])

        # transform
        if mode.lower() == "train":
            self.transform_img = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            )

            self.transform_seg = transforms.Compose(
                [
                    transforms.Resize(img_size),
                ]
            )
        elif mode.lower() == "val":
            self.transform_img = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            )

            self.transform_seg = transforms.Compose(
                [
                    transforms.Resize(img_size),
                ]
            )
        else:
            raise ValueError(
                "Unknown parameter for mode, it should be 'train' or 'val'"
            )

    def __len__(self):
        return len(self.img_address)

    def __getitem__(self, index):
        img = self.transform_img(Image.open(self.img_address[index]))
        seg = self.transform_seg(Image.open(self.seg_address[index]))
        seg = torch.tensor(np.array(seg), dtype=torch.long)
        seg = seg.permute(2, 0, 1)[0, :, :]  # red channel contains class information

        return img, seg


if __name__ == "__main__":
    pass
