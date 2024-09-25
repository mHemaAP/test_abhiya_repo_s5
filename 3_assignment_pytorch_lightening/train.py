import pytorch_lightning as pl
import os
from model import DogBreedModel
from datamodule import DogBreedDataModule

def download_kaggle_dataset():
    # Download the dataset using Kaggle API
    os.system("kaggle datasets download -d khushikhushikhushi/dog-breed-image-dataset -p /data --unzip")

if __name__ == "__main__":
    # Download the dataset from Kaggle
    if not os.path.exists("/data/train"):
        download_kaggle_dataset()

    # Initialize the data module with the dataset location
    dm = DogBreedDataModule(data_dir='/data')
    
    # Initialize the model
    model = DogBreedModel(num_classes=120)

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=10)
    
    # Train the model
    trainer.fit(model, dm)
    
    # Save the model checkpoint
    trainer.save_checkpoint("/app/model/model_checkpoint.ckpt")
