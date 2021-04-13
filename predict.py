import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from preprocessing import save_checkpoint, load_checkpoint, load_cat_names
import json
import os
import random
from PIL import Image
import seaborn as sb

## Defining the parsing arguments for the program

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='5')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/100/image_07897.jpg')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
   
    return parser.parse_args()

def process_image(image):
    '''Same as the notebook
    '''
    img = Image.open(image) # use Image
   
    tansformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_image = tansformer(img)
    
    return processed_image

def predict(image_path, model, topk=5, gpu = 'gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    ## If the machine has gpu 
    if gpu == 'gpu':
        model = model.cuda()
    else:
        model = model.cpu()
    
    
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    if gpu == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)
        
    probability = torch.exp(output.data) 
    probs = np.array(probability.topk(topk)[0][0])
    index_to_class = {val: key for key, val in model.class_to_idx.items()} 
    top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(topk)[1][0])]
    
    return probs, top_classes

def plot(labels,probability):
    base_color = sb.color_palette()[0]
    return sb.barplot(x=probability,y=labels, color = base_color)
    


def main():
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)

    img_path = args.filepath
    probs, classes = predict(img_path, model, int(args.top_k), gpu)
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print('Prediction for the following file: ' + img_path)

    
    i=0
    while i < len(labels):
        print(f"{labels[i]} with a probability of {probability[i]:.3f}")
        i += 1 # cycle through
    
    #plot(labels,probability)
    
if __name__ == "__main__":
    main()

















