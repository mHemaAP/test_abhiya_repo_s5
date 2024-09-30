import os
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Union
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

class DogsDataModule(pl.LightningDataModule):
    def __init__(self, dl_path: str = "data", batch_size: int = 32,num_workers: int = 1):
        super().__init__()
        self._dl_path = Path(dl_path)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = None

    def prepare_data(self):
        """Download images and prepare datasets."""
        dataset_dir = self._dl_path.joinpath("dataset")
        print(f"Checking for dataset in: {dataset_dir}")
        
        # Check if the dataset already exists
        if not dataset_dir.exists():
            download_and_extract_archive(
                url="https://raw.githubusercontent.com/abhiyagupta/Datasets/refs/heads/main/CNN_Datasets/dogs_classifier_dataset.zip",
                download_root=self._dl_path,
                remove_finished=True
            )

    def setup(self, stage: Optional[str] = None):
        """Load data and split into train, val, test sets."""
        print("Setting up data...")
        dataset_df = self.create_dataframe()
        print(f"Total images found: {len(dataset_df)}")

        self.num_classes = dataset_df['label'].nunique()
        print(f"Number of unique classes: {self.num_classes}")
        
        train_df, temp_df = self.split_train_temp(dataset_df)
        val_df, test_df = self.split_val_test(temp_df)

        print(f"Train set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {len(test_df)}")
        

        self.train_dataset = self.create_dataset(train_df)
        self.val_dataset = self.create_dataset(val_df)
        self.test_dataset = self.create_dataset(test_df)

    def create_dataframe(self):
        DATASET_PATH = self._dl_path.joinpath("dataset")
        print(f"Looking for images in: {DATASET_PATH}")
        IMAGE_PATH_LIST = list(DATASET_PATH.glob("*/*.jpg"))
        print(f"Number of images found: {len(IMAGE_PATH_LIST)}")
        images_path = [str(img_path.relative_to(DATASET_PATH)) for img_path in IMAGE_PATH_LIST]
        labels = [img_path.parent.name for img_path in IMAGE_PATH_LIST]
        
        df = pd.DataFrame({'image_path': images_path, 'label': labels})
        print(f"Number of unique labels: {df['label'].nunique()}")
        return df

    def get_num_classes(self):
        if self.num_classes is None:
            raise ValueError("Number of classes is not set. Make sure setup() method is called.")
        return self.num_classes

    def split_train_temp(self, df):
        train_split_idx, temp_split_idx, _, _ = (
            train_test_split(
                df.index, 
                df.label, 
                test_size=0.30,
                stratify=df.label,
                random_state=42
            )
        ) 
        return df.iloc[train_split_idx].reset_index(drop=True), df.iloc[temp_split_idx].reset_index(drop=True)

    def split_val_test(self, df):
        val_split_idx, test_split_idx, _, _ = (
            train_test_split(
                df.index, 
                df.label, 
                test_size=0.5,
                stratify=df.label,
                random_state=42
            )
        )
        return df.iloc[val_split_idx].reset_index(drop=True), df.iloc[test_split_idx].reset_index(drop=True)

    def create_dataset(self, df):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return CustomImageDataset(
            root=str(self._dl_path.joinpath("dataset")),
            image_paths=df['image_path'],
            transform=transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False)


class CustomImageDataset(Dataset):  # Change this line
    def __init__(self, root, image_paths, transform=None):
        self.root = root
        self.image_paths = image_paths
        self.transform = transform
        self.images = []
        self.labels = []

        for idx, path in enumerate(image_paths):
            img = Image.open(os.path.join(root, path))
            self.images.append(img)
            self.labels.append(idx)  # Assuming labels are indices

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        return img, label
