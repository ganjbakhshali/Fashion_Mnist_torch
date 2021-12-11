import os
import numpy as np
import cv2
import torch
import torchvision
from model import *
from argparse import ArgumentParser
import time


parser = ArgumentParser()
parser.add_argument("--device",default="cuda:0", type=str)
parser.add_argument("--input_img",default="1.jpg", type=str)
args = parser.parse_args()

classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
device=torch.device(args.device)
batch_size = 64
epochs = 10
lr = 0.00

transform  = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(32)
])

img = cv2.imread(args.input_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


tensor = transform(img).unsqueeze(0).to(device)


model = mnistF_classifire()
model = model.to(device)

    
model.load_state_dict(torch.load('mnist_Fashion.pth'))
model.eval()
tic= time.time()
pred = model(tensor)
elapsed = time.time() - tic
pred = pred.cpu().detach().numpy()
pred = np.argmax(pred)
output = classes[pred]

print(f"model prediction: {output} \n inference time= {elapsed}")
