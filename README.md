# TensorSlow 2.0

Welcome to my fun little side project!

A couple of years ago Daniel Sabinasz did a great tutorial on the basic functionality of TensorFlow 1 called [TensorSlow](https://github.com/danielsabinasz/TensorSlow). As it was done in pure Python it was also possible to run it with [Pythonista](http://omz-software.com/pythonista/) on iOS. I really liked having the possibility to play around with neural networks on the go to see what happens with different setting. But with TensorFlow 2.0 and mainly using the Keras API for creating the networks, I thought it might be the time to update TensorSlow as well.

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
- chose the Pythonista folder in iCloud
- then under "Repository" choose "Add Remote" and paste the URL to this repository
- Save, fetch and after a couple of seconds you should be able to see the folder in Pythonista


## To Do

A LOT has still to be added! I mean, so far you can just create feedforward networks for MNIST. So here is what to come, when I should have time:
- CIFAR10 / CIFAR100
- CNNs
- Dropout
- Adam and other optimizers
- Other losses and metrics
- ...
