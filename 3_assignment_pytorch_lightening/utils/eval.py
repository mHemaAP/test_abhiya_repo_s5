import pytorch_lightning as pl
from data_module.dogs_datamodule import DogsDataModule
from model.dogs_classifier import DogsClassifier

def main():
    # Load data module
    data_module = DogsDataModule(dl_path='./data', batch_size=32)
    
    # Explicitly call prepare_data and setup
    data_module.prepare_data()
    data_module.setup()

    # Get the number of classes
    num_classes = data_module.get_num_classes()
    print(f"Number of classes in the dataset: {num_classes}")

    # Load the best model using the last checkpoint
    model = DogsClassifier.load_from_checkpoint(
        "checkpoints/dogs_classifier-best_val_loss.ckpt",
        num_classes=num_classes  # Pass the correct number of classes
    )
    
    # Create trainer
    trainer = pl.Trainer(accelerator='auto')
    
    # Evaluate the model
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
