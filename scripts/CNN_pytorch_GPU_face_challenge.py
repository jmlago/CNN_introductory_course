# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import numpy as np
import pandas as pd
import PIL
import os

home = os.path.expanduser("~")

data_path = os.path.join(home,"")

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data

# Hyper Parameters
num_epochs = 15
batch_size = 100
learning_rate = 0.0005

# Convert data into tensors
home = os.path.expanduser("~")
data_folder = os.path.join(home,"Escritorio/face_recognition/fc_data")
os.chdir(data_folder)

# step 1 build pre-data in regular python
filenames = [f for f in os.listdir(data_folder)]
labels = []
for el in filenames:
    if "iniesta" in el:
        labels.append([1,0,0])
    elif "messi" in el:
        labels.append([0,1,0])
    else:
        labels.append([0,0,1])
        
print("\nImage names")
print(filenames[0:32])
print("\nCorresponding Labels")
print(labels[0:32])


## Build training and validation structure
data = pd.DataFrame(filenames,columns=["Names"])
data["Label"] = labels

np.random.seed(2)

T_indexes = np.random.choice(len(filenames),int(0.8*len(filenames)),replace=False)

T_data = data.iloc()[T_indexes]
V_data = data.drop(T_indexes)

T_filenames,T_labels = T_data["Names"].tolist(),T_data["Label"].tolist()
V_filenames,V_labels = V_data["Names"].tolist(),V_data["Label"].tolist()

def _parse_function(filename,label):
    ## doit for each element and convert it to a tensor
    image = PIL.Image.open(filename)
    image = image.convert("RGB")
    image = np.asarray(image, dtype=np.float32) / 255
    image = np.swapaxes(image[:, :, :3],0,2).tolist()
    return image,label

## train prepare
img_tensor = []
lbl_tensor = []
for n,l in zip(T_filenames,T_labels):
    im,lb = _parse_function(n,l)
    img_tensor.append(im)
    lbl_tensor.append(lb)

## val prepare
img_tensor_v = []
lbl_tensor_v = []
for n,l in zip(V_filenames,V_labels):
    im,lb = _parse_function(n,l)
    img_tensor_v.append(im)
    lbl_tensor_v.append(lb)

train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(img_tensor),torch.FloatTensor(lbl_tensor))
val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(img_tensor_v),torch.FloatTensor(lbl_tensor_v))


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(5408, 50)
        self.fc2 = nn.Linear(50, 3)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
        
cnn = CNN()
cnn.cuda()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs.view(-1), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in val_loader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    _, labels = torch.max(labels, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model on the validation images: %d %%' % (100 * correct / total))

# Save the Trained Model
#torch.save(cnn.state_dict(), 'cnn.pkl')