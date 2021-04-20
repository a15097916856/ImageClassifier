#python predict.py flowers/test/1/image_06743.jpg checkpoint_vgg11.pth --top_k 1 --category_names cat_to_name.json --gpu y

# Imports here
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
import json
import argparse

#Define load checkpoint
def load_checkpoint(filepath):
    import torch
    from torchvision import models
    from torch import nn
    checkpoint = torch.load(filepath)
    if filepath == 'checkpoint_vgg11.pth':
        model = models.vgg11(pretrained = True)
    elif filepath == 'checkpoint_vgg19_bn.pth':
        model = models.vgg19_bn(pretrained = True)
    else:
        print('Please choose one of these 2 checkpoints: checkpoint_vgg11.pth, checkpoint_vgg19_bn.pth')
    hidden_units = checkpoint['hidden_units']
    for param in model.parameters():
        param.requires_grad = False
    model.classifier =  nn.Sequential(nn.Linear(25088, hidden_units),
                                      nn.ReLU(),
                                      nn.Dropout(0.4),
                                      nn.Linear(hidden_units, 102),
                                      nn.LogSoftmax(dim = 1))  
    
    model.class_to_idx = checkpoint['index']
    model.load_state_dict(checkpoint['state_dict'])
           
    return model  


    

#Get inputs from user

parser = argparse.ArgumentParser()
parser.add_argument('path_to_image', type = str, help = 'Path to the image that is being used for testing')
parser.add_argument('checkpoint_path', type = str, help = 'Checkpoint that should be used')
parser.add_argument('--top_k', type = int, nargs = "?", help = 'Top prediction categories', default = 1)
parser.add_argument('--category_names', type = str, nargs = "?", const = 'cat_to_name.json', help = 'Category Names', default = "cat_to_name.json")
parser.add_argument('--gpu', type = str, nargs = "?", const = 'y', help = 'Should the model run on gpu when a gpu is available?', default = 'y')

args = parser.parse_args()

def GetUserInputs(args):
    #Get the data directory from command line and build path for training, validation and test datasets
       
    path_to_image = args.path_to_image
    checkpoint_path = args.checkpoint_path
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu
    return path_to_image, checkpoint_path, top_k, category_names, gpu
    
if __name__ == '__main__':
    path_to_image, checkpoint_path, top_k, category_names, gpu = GetUserInputs(args)
         
        
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)        
    
model = load_checkpoint(checkpoint_path)

#run the model on gpu when available, if so chosen by user
if gpu == 'y':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
    
print(device)    

#Define function to process image              
def process_image(path_to_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    norm_mean = np.array([0.485, 0.456, 0.406])
    norm_sd = np.array([0.229, 0.224, 0.225])
    
    image = Image.open(path_to_image)
    
    width, height = image.size
    if width > height:
        height = 256
        ratio = int(width/height)
        width = 256 * ratio
    elif (width == height):
        width = 256
        height = 256
    else:
        width = 256
        ratio = int(height/width)
        height = 256 * ratio
    image = image.resize((width, height))
    image = image.crop(((width - 224)/2, (height - 224)/2, (width + 224)/2, (height + 224)/2)) 
    np_image = np.array(image)
    np_image = np_image / 255
    np_image = (np_image - norm_mean)/norm_sd
    np_image = np_image.transpose((2, 0, 1))
    return np_image
   
              
def predict(path_to_image, checkpoint_path, top_k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = load_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    image = process_image(path_to_image)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze_(0)
    
    inputs = (image_tensor).to(device)
    
    with torch.no_grad():
        logps = model.forward(inputs)
        ps = torch.exp(logps)
        top_p, top_index = ps.topk(top_k, dim = 1)
        top_p = top_p.to('cpu').numpy().tolist()
        top_index = np.array(top_index.cpu())[0]
        index_to_class = {value:key for key, value in model.class_to_idx.items()}
        top_classes = [index_to_class[index] for index in top_index]
        top_flowers = [cat_to_name[classes] for classes in top_classes]
    return top_p, top_classes, top_flowers                    
              
              
probability, classes, flower_names = predict(path_to_image, checkpoint_path, top_k, device)

print(f"Probabilities: {probability}")

print(f"Classes: {classes}")

print(f"Category Names: {flower_names}")
