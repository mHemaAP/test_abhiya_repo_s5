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
import shutil 


class DogsBreadDataModule(pl.LightningDataModule):
    def __init__(
                self,
                batch_size:int,
                num_workers:int,
                pin_memory:bool,
                data_dir:Optional[AnyStr]=None,

    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.prepare_data()

    def prepare_data(self):
        """Download images and prepare datasets."""
        dataset_dir = self._dl_path.joinpath("dataset")
        print(f"Checking for dataset in: {dataset_dir}")

        if not dataset_dir.exists():
            download_and_extract_archive(
                url="https://raw.githubusercontent.com/abhiyagupta/Datasets/refs/heads/main/CNN_Datasets/dogs_classifier_dataset.zip",
                download_root=self._dl_path,
                remove_finished=True
            )

    def setup(self, stage: Optional[str] = None):
        """Load data and split into train, val, test sets."""
        print("Setting up data...")
        
        data_images = self.create_dataframe()
        print(f"Total images found for in dataset- train-test : {len(data_images)}")

        self.num_classes = data_images['label'].nunique()
        print(f"Number of unique classes: {self.num_classes}")

        train_df, test_df = self.split_train_test(data_images)

        print(f"Train set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")

        self.train_dataset = self.create_dataset(train_df)
        self.test_dataset = self.create_dataset(test_df)

        # Handle validation data separately
        val_dir = self._dl_path.joinpath("dataset", "validation")
        if not val_dir.exists():
            val_dir.mkdir(parents=True, exist_ok=True)
            self.prepare_validation_data(train_df)

        # Create input folder for easy access to validation images
        input_folder = Path("input")
        if not input_folder.exists():
            input_folder.mkdir(parents=True, exist_ok=True)
        
        # print validation images number 
        num_val_images = len(list(val_dir.glob("*/*.jpg")))
        print(f"Number of validation images: {num_val_images}")

        # Move validation images to the input folder
        
        for img_path in val_dir.glob("*/*.jpg"):
            dst_path = input_folder.joinpath(img_path.name)
            shutil.copy(img_path, dst_path)

        print(f"Validation images saved to: {input_folder.absolute()}")
        

    def create_dataframe(self):
        DATASET_PATH = self._dl_path.joinpath("dataset")
        
        print(f"Looking for images in: {DATASET_PATH}")
        IMAGE_PATH_LIST = list(DATASET_PATH.glob("*/*.jpg"))
       
        IMAGE_PATH_LIST = [path for path in IMAGE_PATH_LIST if "validation" not in str(path)]
        print(f"Number of images found (excluding validation): {len(IMAGE_PATH_LIST)}")
        images_path = [str(img_path.relative_to(DATASET_PATH)) for img_path in IMAGE_PATH_LIST]
        labels = [img_path.parent.name for img_path in IMAGE_PATH_LIST]

        df = pd.DataFrame({'image_path': images_path, 'label': labels})
        print(f"Number of unique labels: {df['label'].nunique()}")
        print(f"dataframe  is: {df.head()}")
        print(f"shape of dataframe is {df.shape}") 
        return df

    def split_train_test(self, df):
        train_split_idx, test_split_idx, _, _ = train_test_split(
            df.index, df.label, test_size=0.2, stratify=df.label, random_state=self.seed
        )
        return df.iloc[train_split_idx].reset_index(drop=True), df.iloc[test_split_idx].reset_index(drop=True)

    def prepare_validation_data(self, train_df):
        val_split_idx, _, _, _ = train_test_split(
            train_df.index, train_df.label, test_size=0.8, stratify=train_df.label, random_state=self.seed
        )
        val_df = train_df.iloc[val_split_idx].reset_index(drop=True)
        
        for _, row in val_df.iterrows():
            src_path = self._dl_path.joinpath("dataset", row['image_path'])
            dst_path = self._dl_path.joinpath("dataset", "validation", row['label'], src_path.name)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src_path, dst_path)

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

    # def train_dataloader(self):
    #     return DataLoader(self.train_dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True)

    def train_dataloader(self) ->TRAIN_DATALOADERS:
        return DataLoader(
                    dataset=self.train_dataset,
                    batch_size=self.hparams.batch_size,
                    shuffle=True,
                    pin_memory=self.hparams.pin_memory,
                    num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_dataset = ImageFolder(root=str(self._dl_path.joinpath("dataset", "validation")), transform=val_transform)
        return DataLoader(
                    dataset=val_dataset, 
                    batch_size=self.hparams.batch_size,
                    shuffle=False,
                    pin_memory=self.hparams.pin_memory,
                    num_workers=self.hparams.num_workers
        )

     

    # def val_dataloader(self) -> TRAIN_DATALOADERS:
    #     return DataLoader(
    #                 dataset=self.validation_dataset,
    #                 batch_size=self.hparams.batch_size,
    #                 shuffle=False,
    #                 pin_memory=self.hparams.pin_memory,
    #                 num_workers=self.hparams.num_workers
    #     )

    
    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False)


    def test_dataloader(self) -> TRAIN_DATALOADERS:
    return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                pin_memory=self.hparams.pin_memory,
                num_workers=self.hparams.num_workers
    )

class CustomImageDataset(Dataset):
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



# class DogsDataModule(pl.LightningDataModule):
#     def __init__(self, dl_path: str = "data", batch_size: int = 32, num_workers: int = 0):
#         super().__init__()
#         self._dl_path = Path(dl_path)
#         self._batch_size = batch_size
#         self._num_workers = num_workers
#         self.train_dataset = None
#         self.val_dataset = None
#         self.test_dataset = None
#         self.num_classes = None

#     def prepare_data(self):
#         """Download images and prepare datasets."""
#         dataset_dir = self._dl_path.joinpath("dataset")
#         print(f"Checking for dataset in: {dataset_dir}")

#         if not dataset_dir.exists():
#             download_and_extract_archive(
#                 url="https://raw.githubusercontent.com/abhiyagupta/Datasets/refs/heads/main/CNN_Datasets/dogs_classifier_dataset.zip",
#                 download_root=self._dl_path,
#                 remove_finished=True
#             )

#     def setup(self, stage: Optional[str] = None):

#       """Load data and split into train, val, test sets."""
#       print("Setting up data...")
      

#       # Create train and test datasets first
#       train_test_df = self.create_dataframe()
#       print(f"Total images found for train and test: {len(train_test_df)}")

#       self.num_classes = train_test_df['label'].nunique()
#       print(f"Number of unique classes: {self.num_classes}")

#       train_df, test_df = self.split_train_test(train_test_df)

#       print(f"Train set size: {len(train_df)}")
#       print(f"Test set size: {len(test_df)}")

#       self.train_dataset = self.create_dataset(train_df)
#       self.test_dataset = self.create_dataset(test_df)

#       # Handle validation data separately
#       val_dir = self._dl_path.joinpath("dataset", "validation")
#       if not val_dir.exists():
#           val_dir.mkdir(parents=True, exist_ok=True)
#           self.prepare_validation_data(train_df)  # Move validation data

#       # Clear 'input' folder before copying validation images
#       input_folder = Path("input")
#       if input_folder.exists():
#           # Clear input folder if it contains any images
#           for file in input_folder.glob("*"):
#               file.unlink()
#       else:
#           input_folder.mkdir(parents=True, exist_ok=True)

#       # Copy validation images to input folder
#       for img_path in val_dir.glob("*/*.jpg"):
#           dst_path = input_folder.joinpath(img_path.name)
#           shutil.copy(img_path, dst_path)

#       print(f"Validation images saved to: {input_folder.absolute()}")
#       print(f"Number of validation images: {len(list(input_folder.glob('*.jpg')))}")
    
    
#     # Check the number of validation images
 
#     def create_dataframe(self):
#         DATASET_PATH = self._dl_path.joinpath("dataset")
#         print(f"Looking for images in: {DATASET_PATH}")
#         IMAGE_PATH_LIST = list(DATASET_PATH.glob("*/*.jpg"))
#         IMAGE_PATH_LIST = [path for path in IMAGE_PATH_LIST if "validation" not in str(path)]
#         print(f"Number of images found (excluding validation): {len(IMAGE_PATH_LIST)}")
#         images_path = [str(img_path.relative_to(DATASET_PATH)) for img_path in IMAGE_PATH_LIST]
#         labels = [img_path.parent.name for img_path in IMAGE_PATH_LIST]

#         df = pd.DataFrame({'image_path': images_path, 'label': labels})
#         print(f"Number of unique labels: {df['label'].nunique()}")
#         print(f"shape of train test dataframe: {df.head()}") 
#         print(f"shape of train test is {df.shape}")  
#         return df

#     def split_train_test(self, df):
#         train_split_idx, test_split_idx, _, _ = train_test_split(
#             df.index, df.label, test_size=0.2, stratify=df.label, random_state=42
#         )
#         return df.iloc[train_split_idx].reset_index(drop=True), df.iloc[test_split_idx].reset_index(drop=True)

#     def prepare_validation_data(self, train_df):
#         # Split the training data to create validation data
#         val_split_idx, _, _, _ = train_test_split(
#             train_df.index, train_df.label, test_size=0.2, stratify=train_df.label, random_state=42
#         )
#         val_df = train_df.iloc[val_split_idx].reset_index(drop=True)

#         # Move validation images to the validation folder and remove them from training
#         for _, row in val_df.iterrows():
#             src_path = self._dl_path.joinpath("dataset", row['image_path'])
#             dst_path = self._dl_path.joinpath("dataset", "validation", row['label'], src_path.name)
#             dst_path.parent.mkdir(parents=True, exist_ok=True)
#             shutil.move(src_path, dst_path)

#     def create_dataset(self, df):
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#         return CustomImageDataset(
#             root=str(self._dl_path.joinpath("dataset")),
#             image_paths=df['image_path'],
#             transform=transform
#         )

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True)

#     def val_dataloader(self):
#         val_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#         val_dataset = ImageFolder(root=str(self._dl_path.joinpath("dataset", "validation")), transform=val_transform)
#         return DataLoader(val_dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False)
