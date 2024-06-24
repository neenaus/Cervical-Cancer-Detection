import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import datasets,models,transforms
from torch import nn
import os

data_path="D:/base paper/changed mendeley"

new_model=models.resnet18(pretrained=True)

class Net(nn.Module):
  def __init__(self,num_classes):
    super(Net, self).__init__()
    
    self.cnn_layer= torch.nn.Sequential(*(list(new_model.children())[:-1]))
    self.fc=torch.nn.Sequential(nn.Linear(512,num_classes,bias=True),
                                nn.LogSoftmax(dim=1))
    
  def forward(self,x):
    out=self.cnn_layer(x)
    out = out.view(out.size(0),-1)
    x1=out
    out=self.fc(out)
    return x1,out

num_classes= len(os.listdir(data_path))
model=Net(num_classes)

print(model)

data_transforms=transforms.Compose([
                                    transforms.Resize((224,224)),
                                     transforms.ToTensor()
                                     ])

dataset=datasets.ImageFolder(data_path,transform=data_transforms)
data_loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

model.eval()
features=[]
classes=[]
with torch.no_grad():
    for inputs,labels in data_loader:
        x1,output=model.forward(inputs)
        features.append((x1.detach().numpy().tolist())[0])   
        classes.append(labels.numpy().tolist()[0])

df1=pd.DataFrame(features)
df3=pd.DataFrame(classes)
print(df1.shape,df3.shape)