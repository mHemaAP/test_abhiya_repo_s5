import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import lightning as L
from PIL import Image
from torchvision import transforms
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import random

from models.dogs_classifier import DogsBreedClassifier
from utils.logging_utils import setup_logger, task_wrapper, get_rich_progress
from rich.console import Console
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = logging.getLogger(__name__)


@task_wrapper
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return img, transform(img).unsqueeze(0)


@task_wrapper
def infer(model,image_tensor):
    class_labels= ['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd', 'Golden_Retriever','Labrador_Retriever', 'Poodle','Rottweiler','Yorkshire_Terrier']
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor) 
        probabilities = F.softmax(logits,dim=-1)
        y_pred = probabilities.argmax(dim=-1).item()

    confidence = probabilities[0][y_pred].item()
    predicted_label = class_labels[y_pred]
    return predicted_label,confidence

def instantiate_callbacks(callback_cfg):
    callbacks = []
    if callback_cfg:
        for _, cb_conf in callback_cfg.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks

def instantiate_loggers(logger_cfg):
    loggers = []
    if logger_cfg:
        for _, lg_conf in logger_cfg.items():
            if "_target_" in lg_conf:
                loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers

@task_wrapper
def evaluate(trainer, model, datamodule, ckpt_path):
    if ckpt_path:
        # Use the test_dataloader from the datamodule
        test_dataloader = datamodule.test_dataloader()
        trainer.test(model, test_dataloader, ckpt_path=ckpt_path)
    else:
        logging.error("No checkpoint path provided. Cannot evaluate the model.")
        return
    logging.info(f"Evaluation metrics: {trainer.callback_metrics}")


@task_wrapper
def save_prediction_image(image, predicted_label, confidence, output_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


import os
import glob

# def get_latest_checkpoint(base_dir):
#     parent_dir = Path(base_dir)
#     print(f"Looking for checkpoints in parent directory: {parent_dir}")

#     checkpoint_pattern = str(parent_dir / "*" / "checkpoints" / "*.ckpt")
#     print(f"Using checkpoint search pattern: {checkpoint_pattern}")
    
#     checkpoint_files = glob.glob(checkpoint_pattern)
#     print(f"Checkpoint files found: {checkpoint_files}")
    
#     if not checkpoint_files:
#         raise FileNotFoundError(f"No checkpoints found in {parent_dir}")
    
#     latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
#     return latest_checkpoint

# def get_latest_checkpoint(base_dir):
#     parent_dir = Path(base_dir)
#     print(f"Looking for checkpoints in directory: {parent_dir}")

#     checkpoint_pattern = str(parent_dir / "**" / "checkpoints" / "*.ckpt")
#     print(f"Using checkpoint search pattern: {checkpoint_pattern}")
    
#     checkpoint_files = glob.glob(checkpoint_pattern, recursive=True)
#     print(f"Checkpoint files found: {checkpoint_files}")
    
#     if not checkpoint_files:
#         raise FileNotFoundError(f"No checkpoints found in {parent_dir}")
    
#     latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
#     return latest_checkpoint

def get_latest_checkpoint(base_dir):
    base_dir = Path(base_dir)
    print(f"Looking for checkpoints starting from directory: {base_dir}")

    # Start from the base_dir and search upwards until we find a checkpoint
    current_dir = base_dir
    while current_dir != current_dir.parent:  # Stop when we reach the root directory
        checkpoint_pattern = str(current_dir / "**" / "checkpoints" / "*.ckpt")
        print(f"Searching with pattern: {checkpoint_pattern}")
        
        checkpoint_files = glob.glob(checkpoint_pattern, recursive=True)
        if checkpoint_files:
            print(f"Checkpoint files found: {checkpoint_files}")
            return max(checkpoint_files, key=os.path.getctime)
        
        current_dir = current_dir.parent

    raise FileNotFoundError(f"No checkpoints found in or above {base_dir}")

# @hydra.main(version_base="1.3", config_path="../configs", config_name="infer")
# def main(cfg: DictConfig):
#     print(OmegaConf.to_yaml(cfg=cfg))

#     # Set up paths
#     log_dir = Path(cfg.paths.log_dir)
#     setup_logger(log_dir / "infer_log.log")

#     base_dir = cfg.paths.output_dir
#     ckpt_path = get_latest_checkpoint(base_dir)
#     print(f"Using checkpoint: {ckpt_path}")

#     # Load the model with strict=False
#     model = DogsBreedClassifier.load_from_checkpoint(ckpt_path, strict=False)
#     model.eval()

#     input_folder = Path(cfg.input_path)
#     output_folder = Path(cfg.output_path)
#     output_folder.mkdir(exist_ok=True, parents=True)

#     image_files = list(input_folder.glob("*"))
#     selected_images = random.sample(image_files, min(5, len(image_files)))

#     with get_rich_progress() as progress:
#         task = progress.add_task("[green]Processing images...", total=len(selected_images))

#         for image_file in selected_images:
#             if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
#                 img, img_tensor = load_image(image_file)
#                 predicted_label, confidence = infer(model, img_tensor.to(model.device))

#                 output_file = output_folder / f"{image_file.stem}_prediction.png"
            
#                 save_prediction_image(img, predicted_label, confidence, output_file)

#                 progress.console.print(
#                     f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})"
#                 )
#                 progress.advance(task)

# if __name__ == "__main__":
#     main()

# evaluate
# @hydra.main(version_base=None, config_path="../configs", config_name="eval")
# def main(cfg: DictConfig):
#     # Set up paths
#     log_dir = Path(cfg.paths.log_dir)
#     # Set up logger
#     setup_logger(log_dir / "eval_log.log")

#     base_dir = cfg.paths.output_dir
#     ckpt_path = get_latest_checkpoint(base_dir)
#     print(f"Using checkpoint: {ckpt_path}")

#     # Initialize DataModule
#     logging.info("Initializing DataModule")
#     data_module = hydra.utils.instantiate(cfg.data)

#     # Initialize Model
#     logging.info("Initializing Model")
#     model = hydra.utils.instantiate(cfg.model)

#     # Set up callbacks
#     callbacks = instantiate_callbacks(cfg.get("callbacks"))
#     logging.info(f"Initialized {len(callbacks)} callbacks")

#     # Set up loggers
#     loggers = instantiate_loggers(cfg.get("logger"))
#     logging.info(f"Initialized {len(loggers)} loggers")

#     # Initialize Trainer
#     logging.info("Initializing Trainer")
#     trainer = hydra.utils.instantiate(
#         cfg.trainer,
#         callbacks=callbacks,
#         logger=loggers,
#     )

#     # Evaluate the model
#     logging.info("Starting evaluation")
#     if ckpt_path:
#         logging.info(f"Evaluating with model checkpoint: {ckpt_path}")
#         evaluate(trainer, model, data_module, ckpt_path=ckpt_path)
#     else:
#         logging.error("No checkpoint path provided in the configuration.")    
#     # if cfg.ckpt_path:
#     #     logging.info(f"Evaluating with model checkpoint: {cfg.ckpt_path}")
#     #     evaluate(trainer, model, data_module, ckpt_path=cfg.ckpt_path)
#     # else:
#     #     logging.error("No checkpoint path provided in the configuration.")

# if __name__ == "__main__":
#     main()

@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig):
    # Set up paths
    log_dir = Path(cfg.paths.log_dir)
    setup_logger(log_dir / "infer_log.log")

    base_dir = cfg.paths.output_dir
    ckpt_path = get_latest_checkpoint(base_dir)
    log.info(f"Using checkpoint: {ckpt_path}")

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DogsBreedClassifier.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()

    input_folder = Path(cfg.input_path)
    output_folder = Path(cfg.output_path)
    output_folder.mkdir(exist_ok=True, parents=True)

    print(f"input_folder: {input_folder}")
    print(f"output_folder: {output_folder}")

    # Get all image files from the validation folder
    image_files = list(input_folder.glob("**/*.jpg")) + list(input_folder.glob("**/*.png"))
    
    # Randomly select 5 images (or less if there are fewer than 5 images)
    selected_images = random.sample(image_files, min(cfg.num_images, len(image_files)))
    print (len(selected_images))
    console = Console()
    with get_rich_progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(selected_images))

        for image_file in selected_images:
            img, img_tensor = load_image(image_file)
            predicted_label, confidence = infer(model, img_tensor.to(device))

            output_file = output_folder / f"{image_file.stem}_prediction.png"
            save_prediction_image(img, predicted_label, confidence, output_file)
            print(f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})")
            log.info(f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})")
            progress.advance(task)

    log.info(f"Predictions saved to: {output_folder}")

if __name__ == "__main__":
    main()


# def get_latest_checkpoint(base_dir):
#     # Split the base_dir and get the parent directory two levels up
#     parent_dir, _ = os.path.split(base_dir)  # First, remove the last part (17-46-34)
#     # parent_dir, _ = os.path.split(parent_dir)  # Now remove the timestamp to get /code/outputs/2024-10-10

#     # Debugging: Print the directory we are searching in
#     print(f"Looking for checkpoints in parent directory: {parent_dir}")

#     # Define the pattern to search for checkpoints in all subfolders
#     # checkpoint_pattern = os.path.join(parent_dir, "*", "checkpoints", "*.ckpt")
#     checkpoint_pattern = str(Path(parent_dir) / "*" / "checkpoints" / "*.ckpt")    

#     # Debugging: Print the pattern being used for the search
#     print(f"Using checkpoint search pattern: {checkpoint_pattern}")    
    
#     # Use glob to find all checkpoint files matching the pattern
#     checkpoint_files = glob.glob(checkpoint_pattern)

#     # Debugging: Print the checkpoint files found
#     print(f"Checkpoint files found: {checkpoint_files}")    
    
#     if not checkpoint_files:
#         raise FileNotFoundError(f"No checkpoints found in {parent_dir}")
    
#     # Sort the files by creation time and return the latest one
#     latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
#     return latest_checkpoint


# @hydra.main(version_base="1.3", config_path="../configs", config_name="infer")
# def main(cfg: DictConfig):
#     print(OmegaConf.to_yaml(cfg=cfg))

#     # Set up paths
#     log_dir = Path(cfg.paths.log_dir)
#     setup_logger(log_dir / "infer_log.log")

    
#     ckpt_folder = Path(cfg.ckpt_path)

#     # Find the most recent checkpoint file
#     # base_output_dir = Path(cfg.paths.output_dir).resolve()
#     # checkpoint_pattern = str(base_output_dir / "*" / "checkpoints" / "*.ckpt")
#     # checkpoint_files = glob.glob(checkpoint_pattern)
    
#     # if not checkpoint_files:
#     #     raise FileNotFoundError(f"No checkpoint files found matching pattern: {checkpoint_pattern}")
    
#     # # Sort checkpoint files by modification time (most recent first)
#     # most_recent_checkpoint = max(checkpoint_files, key=os.path.getmtime)
#     # print(f"Using checkpoint: {most_recent_checkpoint}")

#     # # Set the environment variable for Hydra to use
#     # os.environ['CHECKPOINT_PATH'] = most_recent_checkpoint
#     # In your infer.py script, before loading the model
#     base_dir = cfg.paths.output_dir  # Path defined in infer.yaml
#     ckpt_path = get_latest_checkpoint(base_dir)

#     # model = DogsBreedClassifier.load_from_checkpoint(most_recent_checkpoint)
#     # model = DogsBreedClassifier.load_from_checkpoint(cfg.ckpt_path)
#     model = DogsBreedClassifier.load_from_checkpoint(ckpt_path, strict=False)    
#     model.eval()

#     input_folder = Path(cfg.input_path)
#     output_folder = Path(cfg.output_path)
#     output_folder.mkdir(exist_ok=True, parents=True)

#     image_files = list(input_folder.glob("*"))
#     selected_images = random.sample(image_files, min(5, len(image_files)))

#     with get_rich_progress() as progress:
#         task = progress.add_task("[green]Processing images...", total=len(image_files))

#         for image_file in selected_images:
#             if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
#                 img, img_tensor = load_image(image_file)
#                 predicted_label, confidence = infer(model, img_tensor.to(model.device))

#                 output_file = output_folder / f"{image_file.stem}_prediction.png"
            
#                 save_prediction_image(img, predicted_label, confidence, output_file)

#                 progress.console.print(
#                     f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})"
#                 )
#                 progress.advance(task)


# @task_wrapper
# def main(args):
#     model = DogsBreedClassifier.load_from_checkpoint(args.ckpt_path)
#     model.eval()

#     input_folder = Path(args.input_folder)
#     output_folder = Path(args.output_folder)
#     output_folder.mkdir(exist_ok=True, parents=True)

#     image_files = list(input_folder.glob("*"))
#     with get_rich_progress() as progress:
#         task = progress.add_task("[green]Processing images...", total=len(image_files))

#         for image_file in image_files:
#             if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
#                 img, img_tensor = load_image(image_file)
#                 predicted_label, confidence = infer(model, img_tensor.to(model.device))

#                 output_file = output_folder / f"{image_file.stem}_prediction.png"
            
#                 save_prediction_image(img, predicted_label, confidence, output_file)

#                 progress.console.print(
#                     f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})"
#                 )
#                 progress.advance(task)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Infer using trained CatDog Classifier"
#     )
#     parser.add_argument(
#         "--input_folder",
#         type=str,
#         required=True,
#         help="Path to input folder containing images",
#     )
#     parser.add_argument(
#         "--output_folder",
#         type=str,
#         required=True,
#         help="Path to output folder for predictions",
#     )
#     parser.add_argument(
#         "--ckpt_path", type=str, required=True, help="Path to model checkpoint"
#     )
#     args = parser.parse_args()

#     log_dir = Path(__file__).resolve().parent.parent / "logs"
#     setup_logger(log_dir / "infer_log.log")

#     main(args)



# if __name__ == "__main__":
#     # parser = argparse.ArgumentParser(
#     #     description="Infer using trained CatDog Classifier"
#     # )
#     # parser.add_argument(
#     #     "--input_folder",
#     #     type=str,
#     #     required=True,
#     #     help="Path to input folder containing images",
#     # )
#     # parser.add_argument(
#     #     "--output_folder",
#     #     type=str,
#     #     required=True,
#     #     help="Path to output folder for predictions",
#     # )
#     # parser.add_argument(
#     #     "--ckpt_path", type=str, required=True, help="Path to model checkpoint"
#     # )
#     # args = parser.parse_args()

#     # log_dir = Path(__file__).resolve().parent.parent / "logs"
#     # setup_logger(log_dir / "infer_log.log")

#     # main(args)
#     main()
