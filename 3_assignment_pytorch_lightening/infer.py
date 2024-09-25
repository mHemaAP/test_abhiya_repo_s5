import pytorch_lightning as pl
from model import DogBreedModel
from PIL import Image
from torchvision import transforms

if __name__ == "__main__":
    # Load model checkpoint
    model = DogBreedModel.load_from_checkpoint("/app/model/model_checkpoint.ckpt")
    model.eval()

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load 10 sample images and run inference
    for i in range(10):
        image = Image.open(f"/data/sample_image_{i}.jpg")
        image = transform(image).unsqueeze(0)
        prediction = model(image)
        print(f"Image {i}: Predicted class {prediction.argmax().item()}")
