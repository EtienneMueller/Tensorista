# Tensorista

Welcome to my fun little side project!

Some time ago I downloaded [Pythonista](http://omz-software.com/pythonista/) to be able to tinker around with little python project when I am on the move. Unfortunately, TensorFlow is not supported and if you want to play around with neural networks, you have to implement them with Numpy by yourself. I did that in the beginning, but really liked the idea to be able to reuse the code I wrote on desktop. So, I started putting everything in the same structure as Keras so it is possible to simply exchange ```import tensorflow as tf``` with ```import tensorista as tf``` and keep the remaining code as is.

So far, it can do everything needed for a very basic neural network:
- Download MNIST and convert it to a numpy array
- Create a sequential model with dense layers
- Print a model summary
- Compile the model
- Train the model
- Evaluate the model

## How to use in Pythonista

There a two ways: the easy one and the better one. The easy one would be to copy the tensorslow folder into site-packages from Pythonista. Then you can import everything like a normal module. But if you want to update, then you have to do it manually (although I might not work on this project that often). The nicer way is to use [Working Copy](https://workingcopy.app/) as git client. As iOS is a bit strict you have to clone the repo the following way:
- press the plus button and chose "Setup synced directory"
- chose the local Pythonista folder
- then under "Repository" choose "Add Remote" and paste the URL to this repository
- Save, fetch and after a couple of seconds you should be able to see the folder in Pythonista


## To Do

There is A LOT I want to add to this project! I mean, so far you can just create feedforward networks for MNIST. So here is what to come, as soon as I find the time:
- CIFAR10 / CIFAR100
- CNNs
- Dropout
- Adam and other optimizers
- Other losses and metrics
- ...
