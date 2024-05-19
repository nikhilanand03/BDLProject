from fastapi import FastAPI, UploadFile, File, Response
from typing_extensions import Annotated
from pydantic import  BaseModel
import torch
from torch import nn
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from typing import Callable
import math
import uvicorn
from efficientnet_pytorch import EfficientNet
import myconfig

from prometheus_fastapi_instrumentator import Instrumentator,metrics
from prometheus_fastapi_instrumentator.metrics import Info

from prometheus_client import Counter, Gauge

import torchvision.transforms as transforms

def api_client_requests_total() -> Callable[[Info], None]:
    api_client_requests_total = Counter(
    'api_client_requests_total',
    'Total number of API requests from each client IP address',
    ['client_ip']
)

    def instrumentation(info: Info) -> None:
        # Increment the counter with the client IP address as a label
        client_ip = info.request.client.host
        api_client_requests_total.labels(client_ip=client_ip).inc()
    return instrumentation

def measure_performance() -> Callable[[Info], None]:
    input_length_gauge = Gauge('input_length', 'Length of the images')
    total_time_gauge = Gauge('total_time', 'Total time taken by the API')
    procdeessing_time_per_char_gauge = Gauge('processing_time_per_char', 'Time per character',['client_ip'])

    async def instrumentation(info: Info) -> None:
        # Increment the counter with the client IP address as a label
        client_ip = info.request.client.host
        length = 784
        input_length_gauge.set(length)
        total_time_gauge.set(info.modified_duration*math.pow(10,6))
        procdeessing_time_per_char_gauge.labels(client_ip=client_ip).set(length/(info.modified_duration*math.pow(10,6)))
    return instrumentation

# path = "/Users/nikhilanand/BDLProject/fastapi_part/b1.pth.tar"
path = "./b1.pth.tar"

def load_model(path: str): # helper function to load the torch model
    checkpoint = torch.load(path,map_location=torch.device('cpu'))
    state_dict = checkpoint["state_dict"]
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(1280, 30)
    model = model.to(myconfig.DEVICE)
    model.load_state_dict(state_dict)
    return model

def transform_image(image_array):
    mean = [0.4897, 0.4897, 0.4897]
    std = [0.2330, 0.2330, 0.2330]

    image_tensor = torch.tensor(image_array, dtype=torch.float32).view(96, 96)

    # Repeat three times in channels
    image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
    print(image_tensor.shape)
    
    # Normalize it
    normalize = transforms.Normalize(mean=mean, std=std)
    image_tensor = normalize(image_tensor)

    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def predict_image(model, image_arr): # helper function that runs the model
    image = transform_image(image_arr)
    image = image.to(myconfig.DEVICE)
    preds = torch.clip(model(image).squeeze(0), 0.0, 96.0)
    print(preds)
    return preds

# def load_image_into_numpy_array(data): # helper function to convert image to np array
#     data1 = BytesIO(data)
#     return np.array(Image.open(data1))

def format_image(data) -> list: # helper function to format the received image (Task 2)
    data1 = BytesIO(data)
    image = Image.open(data1)
    image = image.resize((96,96)).convert("L") # resizes the image to 28*28 pixel and converts to grayscale
    return np.array(image)

app = FastAPI() # initialization of FastAPI module

@app.get("/") # test route to ensure server is running
async def root():
    return {"message": "Hello World"}

@app.post("/predict") # API endpoint for digit prediction, supports POST request
async def predict(image:UploadFile = File(...)): # handler function for the endpoint
    model = load_model(path=path) 
    formatted_image = format_image(await image.read())
    formatted_image = formatted_image.flatten() # serializing into a 1-D array 
    preds = predict_image(model, formatted_image).tolist()
    return {"preds": preds}


Instrumentator().add(measure_performance()).add(api_client_requests_total()).instrument(app).expose(app)
if __name__ =="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)








