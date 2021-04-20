#python train.py flowers --save_dir home/workspace --choose_arch vgg11  --learning_rate 0.01 --hidden_units 512 --epochs 15 --gpu y


import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import figure
import matplotlib.image as mpimg
from collections import OrderedDict
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sb
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type = str, help = 'Data Directory for training, validation and test folders')
parser.add_argument('--save_dir', type = str, nargs = "?", const = 'ImageClassifier', default = 'ImageClassifier', help = 'Location for saving checkpoints')
parser.add_argument('--choose_arch', type = str, nargs = "?", const = 'vgg11', default = 'vgg11', help = 'Choose model architecture - One of vgg11 or vgg19_bn')
parser.add_argument('--learning_rate', type = float, nargs = "?", const = 0.01, default = 0.01, help = 'Learning rate')
parser.add_argument('--hidden_units', type = int, nargs = "?", const = 512, default = 512, help = 'Hidden Units')
parser.add_argument('--epochs', type = int, nargs = "?", const = 15, default = 15, help = '# of epochs')
parser.add_argument('--gpu', type = str, nargs = "?", const = 'y', default = 'y', help = 'Should the model run on gpu when a gpu is available?')

args = parser.parse_args()

def GetUserInputs(args):
    #Get the data directory from command line and build path for training, validation and test datasets
       
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    
    checkpoint_loc = args.save_dir
    arch = args.choose_arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu
    return train_dir, valid_dir, test_dir, checkpoint_loc, arch, learning_rate, hidden_units, epochs, gpu
    
if __name__ == '__main__':
    train_dir, valid_dir, test_dir, checkpoint_loc, arch, learning_rate, hidden_units, epochs, gpu = GetUserInputs(args)
     


    
#transforms for the training, validation, and testing sets
torch.manual_seed(17)
batch_size = 64
epochs = epochs
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

#Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

#Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True) 
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)    



#import the model from torchvision
if arch == 'vgg11':
    model = models.vgg11(pretrained = True)
elif arch == 'vgg19_bn':
    model = models.vgg19_bn(pretrained = True)    
else:
    model = 'Please choose one of these 2 model architectures: vgg11, vgg19_bn'
        

#run the model on gpu when available, if so chosen by user
if gpu == 'y':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
    
print(f"This model will be trained on: {device}")    



#Building the classifier part of the model
for param in model.parameters():
    param.requires_grad = False
    

classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                           nn.ReLU(),
                           nn.Dropout(0.4),
                           nn.Linear(hidden_units, 102),
                           nn.LogSoftmax(dim = 1))
model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr = learning_rate)

model.to(device)



#train the model

epochs = epochs
steps = 0
running_loss = 0
print_every = 1
actual_labels = []

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
                    
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model(inputs)
        loss = criterion(logps,labels)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                   
        

    if steps % print_every == 0:
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                valid_loss += loss.item()
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim = 1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor)).item() 
               
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/print_every: .3f}.. "
              f"Valid loss: {valid_loss/len(validloader): .3f}.. "
              f"Accuracy: {accuracy/len(validloader)*100: .3f}"
              )         
        running_loss = 0
        model.train()
 

# Validation of trained model on the test set

print('Testing the trained model')

step = 0
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        step += 1
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model(inputs)
        loss = criterion(logps, labels)
        test_loss += loss.item()
        
        #Accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item() 
                
        print(f"Step {step}.. "
              f"Test loss: {test_loss/len(testloader): .3f}.. "
              f"Testing Accuracy: {accuracy/len(testloader)*100: .3f}"
              )         
        



#save checkpoint

checkpoint = {
              'state_dict' : model.state_dict(),
              'index' : train_dataset.class_to_idx,
              'hidden_units' : hidden_units
              }
checkpoint_path = 'checkpoint_'+ arch + '.pth'
torch.save(checkpoint, checkpoint_path)
print(f"Checkpoint saved as: {checkpoint_path}")







