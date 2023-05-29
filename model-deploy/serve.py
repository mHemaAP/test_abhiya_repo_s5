from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
from PIL import Image
import torch
from io import BytesIO
from typing import List
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential, ReLU, LogSoftmax, Flatten

app = FastAPI()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = Sequential(
            Conv2d(1, 32, 3, 1),
            ReLU(),
            Conv2d(32, 64, 3, 1),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(9216, 128),
            ReLU(),
            Linear(128, 10),
            LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.main(x)

# Load the model
model = Net()
model.load_state_dict(torch.load("./model/mnist_cnn.pt"))
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        image_data = await file.read()
        # print(image_data)
        image = Image.open(BytesIO(image_data)).convert('L')
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")

    with torch.no_grad():
        prediction = model(image)
        predicted_class = torch.argmax(prediction, dim=1)

    return {"class": int(predicted_class)}
