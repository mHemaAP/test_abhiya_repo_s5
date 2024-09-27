import pytorch_lightning as pl
from data_modules.dogs_datamodule import DogsDataModule
from models.dogs_classifier import DogsClassifier

def main():
    # Load data module
    data_module = DogsDataModule(data_dir='data', batch_size=32)
    
    # Load the best model using the last checkpoint
    model = DogsClassifier.load_from_checkpoint("checkpoints/best_model.ckpt")
    
    # Create trainer
    trainer = pl.Trainer(accelerator='auto')
    
    # Evaluate the model
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()

    