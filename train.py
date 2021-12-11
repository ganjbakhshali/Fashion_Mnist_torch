import torch  
import torchvision
from model import *

from argparse import ArgumentParser



# data_transform=torchvision.transforms.Compose(
#     [torchvision.transforms.ToTensor(),
#      torchvision.transforms.Normalize((0),(1))]
# )

# dataset=torchvision.datasets.FashionMNIST("./dataset",train=True,download=True,transform=data_transform)

def get_fashion_mnist_dataloaders(batch_size=128):
    """Fashion MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor()
    ])
    # Get train and test data
    train_data = torchvision.datasets.FashionMNIST('./fashion_mnist_train', train=True, download=True,
                                       transform=all_transforms)
    test_data = torchvision.datasets.FashionMNIST('./fashion_mnist_test', train=False,download=True,
                                      transform=all_transforms)
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader 

def cal_acc(y_hat,labels):
    _,y_hat_max=torch.max(y_hat,1)
    acc=torch.sum(y_hat_max==labels.data,dtype=torch.float64)/len(y_hat)
    return acc


if __name__=="__main__":

    # device=torch.device("cuda")
    parser = ArgumentParser()
    parser.add_argument("--device",default="cuda:0", type=str)
    args = parser.parse_args()
    device=torch.device(args.device)

    model=mnistF_classifire()
    model=model.to(device)
    model.train(True)

    batch=64
    epoch=10
    lr=0.1
    optimizer=torch.optim.SGD(model.parameters(),lr=lr)
    loss_func=torch.nn.CrossEntropyLoss()
    train_loader, test_loader=get_fashion_mnist_dataloaders(batch)

    #train
for ep in range(epoch):
    train_loss=0.0
    train_acc=0.0

    for im,labels in train_loader:
        im=im.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()

        #forwarding
        y_hat=model(im)

        #backwarding
        loss=loss_func(y_hat,labels)
        loss.backward()

        #update
        optimizer.step()


        train_loss+=loss
        train_acc+=cal_acc(y_hat,labels)


    total_loss=train_loss/len(train_loader)
    total_acc=train_acc/len(train_loader)
    print(f"epoch:{ep} , Loss:{total_loss} , accuracy: {total_acc}")

torch.save(model.state_dict(),"mnist_Fashion.pth")