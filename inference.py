import torch  
import torchvision
from model import *


device=torch.device("cuda")
model=mnistF_classifire()
model=model.to(device)
model.load_state_dict(torch.load('mnist_Fashion.pth'))
model.eval()

