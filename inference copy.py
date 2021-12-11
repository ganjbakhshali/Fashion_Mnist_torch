import torch  
import torchvision
from model import *
import cv2
import numpy as np
from PIL import Image

device=torch.device("cuda")
model=mnistF_classifire()
model=model.to(device)
model.load_state_dict(torch.load('mnist_Fashion.pth'))
model.eval()

all_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.ToTensor()
])

# img=cv2.imread("1.jpg")
im = Image.open('1.jpg')
newsize = (32, 32)
im = im.resize(newsize)
# img = np.asarray(im)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img=cv2.resize(img,(32,32))
tensor=all_transforms(im).unsqueeze(0).to(device)

y_hat=model(tensor)

y_hat=y_hat.cpu().detach().numpy()
output=np.argmax(y_hat)
output

