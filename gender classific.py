#!/usr/bin/env python
# coding: utf-8

# # Gender Classification

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.nn.functional as F
from torchvision import datasets


# # Model constrcution

# In[39]:


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1)
        self.conv2=torch.nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.pooling=torch.nn.MaxPool2d(2)
        self.fc1=torch.nn.Linear(32*56*56,256)
        self.fc2=torch.nn.Linear(256,2)
    def forward(self,x):
        batch_size=x.size(0)
        x=F.relu(self.pooling(self.conv1(x)))
        x=F.relu(self.pooling(self.conv2(x)))
        x=x.view(batch_size,-1)
        x=self.fc1(x)
        x=self.fc2(x)
        return x


# # Data processing

# In[40]:


batch_size=64
transform=transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_dataset=datasets.ImageFolder(root='Desktop/Training',transform=transform)
train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset=datasets.ImageFolder(root='Desktop/Test',transform=transform)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
train_num=len(train_dataset)


# # Train Model

# In[41]:


model=Net()
device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model.to(device)
loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.0002)

epochs=10
save_path='Desktop/Net.pth'
def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        inputs,target=inputs.to(device),target.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=loss_function(outputs,target)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if batch_idx%300==299:
            print('[%d,%5d]loss:%.3f'%(epoch+1,batch_idx+1,running_loss))


# # Test Model

# In[42]:


#This is test code
def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            _,prediceted=torch.max(outputs.data,dim=1)
            total+=labels.size(0)
            correct+=(prediceted==labels).sum().item()
    print('Accuracy on test set:%d %%'%(100*correct/total))


# # Performance

# In[43]:


if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
        test()


# In[ ]:




