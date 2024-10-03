import torch
from torchvision import transforms
from model.dogs_classifier import DogsClassifier
import os
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from pathlib import Path

def inference(model, image_path, output_path):
    # Load and preprocess the image
    img = PILImage.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transform to the image
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Move the input tensor to the same device as the model
    img_tensor = img_tensor.to(model.device)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Get class labels (you need to provide these based on your specific dog breeds)
    class_labels = ["Beagle", "Boxer", "BullDog", "Dachshund", "German Shepard", 
                    "Golder Retriever", "Labrador Retriver", "Poodle", "Rottweilier", "Yorkshire Terrier"]  
    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()

    # Visualize the prediction
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {predicted_label.capitalize()} (Confidence: {confidence:.2f})')
    
    # Display the figure
    plt.show()
    
    # Save the figure to the output folder
    plt.savefig(output_path)
    plt.close()

    return predicted_label, confidence

def main():
    # Set up input and output folders
    input_folder = Path("input")
    output_folder = Path("output")
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True)

    # Load the model
    checkpoint_path = '/app/checkpoints/dogs_classifier-best_val_loss.ckpt'
    model = DogsClassifier.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()

    # Get all image files from the input folder
    image_files = list(input_folder.glob('*.jpg')) + list(input_folder.glob('*.png'))

    # Randomly select 5 images (or all if less than 5)
    selected_images = random.sample(image_files, min(5, len(image_files)))

    # Process each selected image
    for img_path in selected_images:
        output_path = output_folder / f"{img_path.stem}_prediction.png"
        predicted_label, confidence = inference(model, img_path, output_path)
        print(f"Processed {img_path.name}: Predicted {predicted_label} with confidence {confidence:.2f}")
        print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()





