import pytorch_lightning as pl
from data_module.dogs_datamodule import DogsDataModule
from model.dogs_classifier import DogsClassifier

def main():
    # Load data module
    data_module = DogsDataModule(dl_path='data', batch_size=32)  # Fixed dl_path
    
    # Load the best model using the last checkpoint
    model = DogsClassifier.load_from_checkpoint("checkpoints/dogs_classifier-epoch=XX-val_loss=XX.ckpt")
    
    # Create trainer
    trainer = pl.Trainer(accelerator='auto')
    
    # Evaluate the model
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
