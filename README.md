# Image-Classifier
In this project, the main objective is to train a neural netowrk to learn how to classify flowers, and thereafter recognize and correctly classify any image provided, obviously, the dataset is for 102 different types of flowers https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

When you've completed this project, you'll have an application that can be trained on any set of labelled images. Here your network will be learning about flowers and end up as a command line application. 

## Prerequisites
1. The Code is written in **Python 3.6.5** . If you don't have Python installed you can find it [here](https://www.python.org/downloads/).
Ensure you have the latest version of pip.
2. Additional Packages that are required are: [Numpy](http://www.numpy.org/), [Pandas](https://pandas.pydata.org/), [MatplotLib](https://matplotlib.org/), [Pytorch](https://pytorch.org/), PIL and json. You can donwload them using [pip](https://pypi.org/project/pip/):
    - ```pip install numpy pandas matplotlib pil```<br/>
    or [conda](http://www.numpy.org/)
    - ```conda install numpy pandas matplotlib pil```
    
**NOTE**: In order to install Pytorch follow the instructions given on the official site.

## Command Line Application

- Train a new network on a data set with **train.py**
  - Basic Usage : ```python train.py data_directory```<br/>
  - Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  - Options:
    - Set direcotry to save checkpoints: ```python train.py data_dor --save_dir save_directory```
    - Choose arcitecture (vgg13 or vgg16, vgg16 is the default): ```python train.py data_dir --arch "vgg16"```
    - Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --hidden_layer1 500 --epochs 20```
    - Use GPU for training: ```python train.py data_dir --gpu gpu```
  - Output: A trained network ready with checkpoint saved for doing parsing of flower images and identifying the species.
    
- Predict flower name from an image with **predict.py** along with the probability of that name. That is you'll pass in a single image /path/to/image and return the flower name and class probability
  - Basic usage: ```python predict.py /path/to/image checkpoint```
  - Options:
    - Return top K most likely classes: ```python predict.py input checkpoint ---top_k 5```
    - Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_To_name.json```
    - Use GPU for inference: ```python predict.py input checkpoint --gpu```

## Data
The data used specifically for this assignment are a flower database(.json file). It is not provided in the repository as it's larger than what github allows.<br/>
The data need to comprised of 3 folders:
1. test
2. train 
3. validate<br/>

Unfurtunetly the data is not provided 

## GPU/CPU
Please note that GPU is recommended for Deep Neural Network

## Hyperparameters
The hyperparameters were tested extensively, the training time using CPU is around 10 minutes with 3 epochs

- By increasing the number of epochs the accuracy of the network on the training set gets better, however, it is worth highlighting that this follows the law of dimishing returns, and the accuracy won't justify the additional time for training your model
- A big learning rate guarantees that the network will converge fast to a small error but it will constantly overshot
- A small learning rate guarantees that the network will reach greater accuracies but the learning process will take longer
- Both vgg13 and vgg16 provide 95% accuracy at least in the tests performed prior submission

Current settings:<br/>
- lr: 0.001
- dropout: 0.2
- epochs: 3

## Final Results - Up to 95% success

Unfurtunetly we cannot upload the dataset, so you won't be able to run the code an obtain a result

