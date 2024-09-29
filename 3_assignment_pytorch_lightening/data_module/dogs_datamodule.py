 
import os
import zipfile
from pathlib import Path
from typing import Optional, Union
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive



class DogsDataModule(pl.LightningDataModule):
    def __init__(self, dl_path: Union[str, Path] = "data", num_workers: int = 0, batch_size: int = 8):
        super().__init__()
        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size

    def prepare_data(self):
        """Download images and prepare images datasets."""
        download_and_extract_archive(
            url="https://raw.githubusercontent.com/abhiyagupta/Datasets/refs/heads/main/CNN_Datasets/dogs_classifier_dataset.zip",
            download_root=self._dl_path,
            remove_finished=True
        )

    @property
    def data_path(self):
        return Path(self._dl_path).joinpath("dogs_classifier_dataset")

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def valid_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def __dataloader(self, train: bool):
        """Train/validation/test loaders."""
        if train:
            dataset = self.create_dataset(self.data_path.joinpath("train"), self.train_transform)
        else:
            dataset = self.create_dataset(self.data_path.joinpath("validation"), self.valid_transform)
        return DataLoader(dataset=dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=train)

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)

    def test_dataloader(self):
        return self.__dataloader(train=False)  # Using validation dataset for testing






# =================================

# class DogsDataModule(pl.LightningDataModule):
#     def __init__(self, dl_path: Union[str, Path] = "data", num_workers: int = 0, batch_size: int = 8):
#         super().__init__()
#         self._dl_path = dl_path
#         self._num_workers = num_workers
#         self._batch_size = batch_size

#     def prepare_data(self):
#         """Download images and prepare images datasets."""
#         download_and_extract_archive(
#             url="https://github.com/abhiyagupta/Datasets/raw/main/CNN_Datasets/dogs_classifier_dataset.zip",
#             download_root=self._dl_path,
#             remove_finished=True
#         )
        
#         # Extract dataset into separate directories for training and validation
#         os.makedirs(os.path.join(self._dl_path, "train"), exist_ok=True)
#         os.makedirs(os.path.join(self._dl_path, "validation"), exist_ok=True)
        
#         # Move images to train directory
#         for file in os.listdir(os.path.join(self._dl_path, "dogs_classifier_dataset")):
#             if file.endswith('.jpg') or file.endswith('.png'):
#                 os.rename(os.path.join(self._dl_path, "dogs_classifier_dataset", file),
#                           os.path.join(self._dl_path, "train", file))

#     def data_path(self):
#         return Path(self._dl_path)

#     def normalize_transform(self):
#         return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#     def train_transform(self):
#         return transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             self.normalize_transform(),
#         ])

#     def valid_transform(self):
#         return transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             self.normalize_transform()
#         ])

#     def create_dataset(self, data_dir, transform):
#         return ImageFolder(root=data_dir, transform=transform)

#     def __dataloader(self, train: bool):
#         """Train/validation loaders."""
#         if train:
#             dataset = self.create_dataset(self.data_path().joinpath("train"), self.train_transform())
#         else:
#             dataset = self.create_dataset(self.data_path().joinpath("validation"), self.valid_transform())
#         return DataLoader(dataset=dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=train)

#     def train_dataloader(self):
#         return self.__dataloader(train=True)

#     def val_dataloader(self):
#         return self.__dataloader(train=False)



