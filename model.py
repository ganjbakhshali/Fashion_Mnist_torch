import torch




class mnistF_classifire(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.L1=torch.nn.Linear(1024,128)
        self.L2=torch.nn.Linear(128,10)


    def forward(self,x):
        #input batch shape 32*28*28
        x=x.reshape((x.shape[0],1024))
        #x will be 32*784
        z=self.L1(x)
        z=torch.relu(z)
        z=torch.dropout(z,0.2,train=True)
        z=self.L2(z)
        y=torch.softmax(z,dim=1)
        return y