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
from preprocessing import save_checkpoint, load_checkpoint
import json
import os
import random

## Defining the parsing arguments for the program

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg13', 'vgg16'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='500')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

def train(model,criterion,optimizer,trainloader,epochs,gpu):
    steps = 0
    print_every = 5    

    for epoch in range(epochs):
        running_loss = 0
        for i,(inputs,labels) in enumerate(trainloader[0]):
            steps +=1
            # The following code check if a GPU machine is available
            if gpu == 'gpu':
                model.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda') # use cuda
            else:
                model.cpu() 
            optimizer.zero_grad() # Cleans the gradient for each iteration
            #Forward Model
            logps = model.forward(inputs)
            # Calculate a loss function, meanning the difference between label and logps
            loss = criterion(logps, labels)
            # Propagate the model backwards (backpropagation)
            loss.backward()
            # Optimize the weights in the network
            optimizer.step()
            # add all the losses
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval() ## THis allows for droput during the model evaluation 
                
                for ii, (inputs2,labels2) in enumerate(trainloader[2]):  # Please note that trainloader is a list with 3 dataset images, trainloader[2] is validation dataset
                    if gpu == 'gpu':
                        inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda')
                        model.to('cuda:0')
                    else:
                        pass
                      
                    with torch.no_grad():
                        logps = model.forward(inputs2)
                        batch_loss = criterion(logps, labels2)
                        test_loss += batch_loss.item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels2.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()                  
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Accuracy loss: {test_loss/len(trainloader[2]):.3f}.. "
                          f"Accuracy: {accuracy/len(trainloader[2]):.3f}")
                    running_loss = 0

def main():
    print("Training is about to start, please note that this is going to take some time")
    args = parse_args()
    
    #As per the example in the notebooks
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
   
    
    
    # DONE: Obviously you can have a different transform for validation, but in essence is the same, so no need to another transformation
    image_data = [ ImageFolder(train_dir, transform=train_transform), 
                   ImageFolder(test_dir, transform=test_transform), 
                   ImageFolder(val_dir, transform=test_transform)]
    


    trainloader = [torch.utils.data.DataLoader(image_data[0], batch_size=64, shuffle = True),
                   torch.utils.data.DataLoader(image_data[1], batch_size=64, shuffle = True),
                   torch.utils.data.DataLoader(image_data[2], batch_size=64, shuffle = True)]
    
    model = getattr(models, args.arch)(pretrained=True)
    
    #Freezes the parameters for the pretrained model
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == "vgg16":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 500)),
                                  ('drop', nn.Dropout(p=0.2)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
              
    elif args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(1024, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
                   
      
    model.classifier = classifier 
    # Defining the loss function
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = image_data[0].class_to_idx               
    gpu = args.gpu              
    train(model, criterion, optimizer, trainloader, epochs, gpu)              
    model.class_to_idx = class_index
    path = args.save_dir
    save_checkpoint(path, model, optimizer, args, classifier)

if __name__ == "__main__":
    main()






