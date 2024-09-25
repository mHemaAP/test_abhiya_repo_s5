import os
import json
import pytorch_lightning as pl
from model import DogBreedModel  # Ensure this is your model's actual class name
from datamodule import DogBreedDataModule  # Ensure this is your DataModule's actual class name

def main():
    # Path to the model checkpoint
    checkpoint_path = "/app/model/model_checkpoint.ckpt"  # Modify as necessary
    results_file = "./model/eval_results.json"  # File to store evaluation results

    # Check if the checkpoint file exists
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Evaluation cannot proceed.")
        return

    # Load the model from the checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = DogBreedModel.load_from_checkpoint(checkpoint_path)

    # Initialize the data module
    dm = DogBreedDataModule(data_dir='/data')  # Adjust path as necessary

    # Initialize the Trainer
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0)

    # Evaluate the model
    print("Evaluating model...")
    results = trainer.validate(model, datamodule=dm)

    # Print and save the evaluation metrics
    print("Evaluation Results:")
    for result in results:
        for key, value in result.items():
            print(f"{key}: {value}")

    # Save the results to a JSON file
    with open(results_file, 'w') as f:
        json.dump(results, f)

    # Check if the results were saved successfully
    if os.path.isfile(results_file):
        print(f"Evaluation completed successfully. Results saved at {results_file}.")
    else:
        print("Evaluation did not complete successfully.")

if __name__ == "__main__":
    main()
