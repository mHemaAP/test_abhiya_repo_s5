import requests
import torch
import random
from torchvision import datasets
from PIL import Image
from io import BytesIO
import base64
import os
import concurrent.futures
import time

# Create requests directory if it doesn't exist
os.makedirs('requests', exist_ok=True)

# Load MNIST dataset
dataset = datasets.MNIST('./data', train=False, download=True)

def make_request(index):
    # Select a random image from the dataset for testing
    image, label = dataset[index]

    # Save the image in a buffer
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)

    # Encode the image in base64 and send a POST request
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    response = requests.post('http://localhost:8000/predict/', files={'file': ("image.png", BytesIO(base64.b64decode(img_str)), 'image/png')})

    # print(f'Index: {index}, Label: {label}, Prediction: {response.json()["class"]}')

    # Save the image to the requests directory
    filename = f'requests/predicted_{response.json()["class"]}_label_{label}_index_{index}.png'
    with open(filename, 'wb') as f:
        f.write(base64.b64decode(img_str))

    return index

# Number of requests to make in the load test
num_requests = 1_000

start_time = time.time()

# Use a ThreadPoolExecutor to perform the requests in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
    indices = random.sample(range(len(dataset)), num_requests)
    futures = [executor.submit(make_request, index) for index in indices]

    for future in concurrent.futures.as_completed(futures):
        print(f'Completed request {future.result()}')

end_time = time.time()
duration = end_time - start_time
requests_per_second = num_requests / duration

print(f'Completed {num_requests} in {duration} seconds')
print(f'{requests_per_second} requests per second')
