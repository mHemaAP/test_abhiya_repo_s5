import os
import zipfile
import requests
from pathlib import Path
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

class DogsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'data', batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.dataset_url = "https://github.com/abhiyagupta/Datasets/raw/main/CNN_Datasets/dogs_classifier_dataset.zip"
        self.dataset_zip_path = os.path.join(self.data_dir, 'dogs_classifier_dataset.zip')

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir, 'dogs_classifier_dataset')):
            self.download_and_extract_dataset()

    def download_and_extract_dataset(self):
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(self.dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url}...")
            try:
                response = requests.get(self.dataset_url)
                response.raise_for_status()  # Raises an HTTPError for bad responses
                with open(self.dataset_zip_path, 'wb') as f:
                    f.write(response.content)
                print("Download complete!")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading the dataset: {e}")
                return

        print("Extracting dataset...")
        try:
            with zipfile.ZipFile(self.dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("Extraction complete!")
        except zipfile.BadZipFile:
            print("Error: The downloaded file is not a valid zip file.")
            # Optionally, delete the invalid zip file
            os.remove(self.dataset_zip_path)
        except Exception as e:
            print(f"Error extracting the dataset: {e}")


    def setup(self, stage: Optional[str] = None):
        # Ensure the dataset is ready
        self.prepare_data()

        # Load the data
        dataset_path = os.path.join(self.data_dir, 'dogs_classifier_dataset')
        full_dataset = ImageFolder(dataset_path, transform=self.transform)

        # Split the dataset into training, validation, and test sets
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
