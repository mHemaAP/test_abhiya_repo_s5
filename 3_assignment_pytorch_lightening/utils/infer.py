import torch
from PIL import Image
from torchvision import transforms
from model.dogs_classifier import DogsClassifier

def main():
    # Load the best model using the last checkpoint
    model = DogsClassifier.load_from_checkpoint("checkpoints/dogs_classifier-best_val_loss.ckpt")
    model.eval()
    
    # Prepare image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image_path = input("Enter the path to your image: ")
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1)
    
    print(f"Predicted class: {prediction.item()}")

if __name__ == "__main__":
    main()
