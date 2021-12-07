import torch
import torchvision
from model import *
# from train import cal_acc


def cal_acc(y_hat,labels):
    _,y_hat_max=torch.max(y_hat,1)
    acc=torch.sum(y_hat_max==labels.data,dtype=torch.float64)/len(y_hat)
    return acc

all_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.ToTensor()
])

batch_size=100
test_data = torchvision.datasets.FashionMNIST('./fashion_mnist_test', train=False,download=True,
                                    transform=all_transforms)
# Create dataloaders

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
device=torch.device("cuda")
model=mnistF_classifire()
model=model.to(device)
model.load_state_dict(torch.load('mnist_Fashion.pth'))
model.eval()



#test
test_acc=0.0
for im,labels in test_loader:
    im=im.to(device)
    labels=labels.to(device)

    #forwarding
    y_hat=model(im)


    test_acc+=cal_acc(y_hat,labels)



total_acc=test_acc/len(test_loader)
print(f"accuracy: {total_acc}")