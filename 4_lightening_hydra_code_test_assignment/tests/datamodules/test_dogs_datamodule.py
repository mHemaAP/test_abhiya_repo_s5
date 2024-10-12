import pytest
import torch 
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.data_modules.dogs_datamodule import DogsBreedDataModule




import os 
@pytest.fixture
def datamodule():
    print(os.getcwd())
    
    dm = DogsBreedDataModule(batch_size=8,
                num_workers=0,
                pin_memory=False,
                data_dir="./data/dogs_dataset")
    print(dm)
    return dm



def test_dogsbreed_data_setup(datamodule):
    datamodule.prepare_data()
    datamodule.setup()

    assert datamodule.transforms is not None
    assert datamodule.train_dataset is not None
    assert datamodule.test_dataset is not None
    assert datamodule.val_dataset is not None



def test_prepare_data(datamodule):
    datamodule.prepare_data()
    datamodule.setup()
    assert len(datamodule.train_dataset) > len(datamodule.test_dataset)
    assert len(datamodule.train_dataset) > len(datamodule.val_dataset)


def test_dogsbreed_dataloaders(datamodule):
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

if __name__=='__main__':
    datamodule()