import os
import json
from typing import List, Tuple
from PIL import Image
from PIL.Image import NEAREST
from PIL.ImageStat import Stat
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from enum import Enum, unique
import random


BASE_IMAGE_DIR = "/ssd/datasets/chest_xray"
DIMS = (256, 256)


def train_composer(dims, stddev, mean):
    return transforms.Compose(
        [
            transforms.Resize(dims, interpolation=NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=1.5),
            transforms.ToTensor(),  # also divides by 255
            transforms.Normalize(stddev, mean),
        ]
    )


def inference_composer(dims, stddev, mean):
    return transforms.Compose(
        [
            transforms.Resize(dims, interpolation=NEAREST),
            transforms.ToTensor(),  # also divides by 255
            transforms.Normalize(stddev, mean),
        ]
    )


@unique
class SetType(Enum):
    test = 1
    train = 2
    val = 3


class PneumoniaDataset(Dataset):
    def __init__(self, set_type: SetType, shuffle=True):
        self.set_type = set_type
        self.base_dir = BASE_IMAGE_DIR
        self.shuffle = shuffle
        self.metadata = self._load_metadata()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        image = Image.open(
            os.path.join(self.base_dir, self.metadata[idx]["image"])
        ).convert("L")
        stats = Stat(image)
        stddev = np.divide(stats.stddev, 255)
        mean = np.divide(stats.mean, 255)

        curr_img_label = self.metadata[idx]["label"]
        random_idx = random.randint(0, len(self) - 1)
        diff_img_label = self.metadata[random_idx]["label"]
        get_same_class = random.choice([True, False])
        if get_same_class:
            while curr_img_label != diff_img_label:
                random_idx = random.randint(0, len(self) - 1)
                diff_img_label = self.metadata[random_idx]["label"]
        else:
            while curr_img_label == diff_img_label:
                random_idx = random.randint(0, len(self) - 1)
                diff_img_label = self.metadata[random_idx]["label"]

        distance = 1 if get_same_class else 0

        image2 = Image.open(
            os.path.join(self.base_dir, self.metadata[random_idx]["image"])
        ).convert("L")
        stats2 = Stat(image2)
        stddev2 = np.divide(stats2.stddev, 255)
        mean2 = np.divide(stats2.mean, 255)

        if self.set_type == SetType.train:
            composer = train_composer(DIMS, stddev, mean)
            composer2 = train_composer(DIMS, stddev2, mean2)
        else:
            composer = inference_composer(DIMS, stddev, mean)
            composer2 = inference_composer(DIMS, stddev2, mean2)
        return composer(image), composer2(image2), distance

    def _load_metadata(self) -> List[dict]:
        metadata_path = "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        data = [doc for doc in metadata if doc["set"] == self.set_type.name]
        if self.shuffle:
            random.shuffle(data)
        return data
