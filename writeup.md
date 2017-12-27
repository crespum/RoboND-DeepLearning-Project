# Project: Follow Me

[//]: # (Image References)
[fcn_architecture]: ./misc/fcn_architecture.png

Here I will consider the [rubric](https://review.udacity.com/#!/rubrics/1155/view) points individually and describe how I addressed each point in my implementation.  

## A. Writeup / README

* Provide a write-up / README document including all rubric items addressed in a clear and concise manner. The document can be submitted either in either Markdown or a PDF format.

You are reading it!

* The write-up conveys the an understanding of the network architecture.

The architecture is based on a Fully Convolutional Network, whose main advantage is the ability to retain spatial information. It's made up of 3 main blocks: an encoder, a decoder and a 1x1 convolution:

The **encoder** extracts the main features from an image whereas the **decoder** maps them to the original image. Each layer of the encoder used in this project consists of a separable convolutional layer and a batch normalization with an activation function included. The decoder consists of a bilinear upsampling layer, a concatenation layer and a separable convolutional layer.

Each layer of the encoder has a clear function. The **separable convolutional layer** reduces the number of parameters of the network and identifies the main features (extracts spatial information). The **batch normalization layer**, as the name implies, normalizes the inputs to each layer within the network to have a well conditioned problem. The **activation function** (a RELU in this case) adds non-linearities to the network for it to better fit the model.

On the decoder side, the **bilinear upsampling layer** expands the dimensions of the encoded information (the objective is to reach the size of the original image). The **concatenation layer** is similar to skipping the connections, which is a technique that allows one layer to use information from different resolution scales (from layers prior to the preceding one). Finally, the last piece of each decoder layer is another separable convolutional layer.

Between the decoder and the encoder, there is a **1x1 convolution layer** that increases the depth of the network while preserving spatial information. This is very common for object detection and semantic segmentation. Additionally, it has the advantage that during inference we can feed images of any size into the trained network. The alternative would be a fully-connected layer, which keeps the number of features but needs a fixed image size.

![Network architecture][fcn_architecture]

* The write-up conveys the student's understanding of the parameters chosen for the the neural network.

The final parameters chosen for the network are:

```
learning_rate = 0.001
batch_size = 32
num_epochs = 20
steps_per_epoch = 200
validation_steps = 50
workers = 4
```

In the first attempt, I chose a `learning rate` of 0.1 and `num_epochs` of 5 because those are typically reasonable parameters for testing the network in a local environment. After validating the implementation (both the training loss and the validation loss were decreasing but the performance was not good enough), I ported the environment to Amazon AWS and trained the network decreasing the learning rate and increasing the number of epochs until reaching the final parameters, which yield an **IoU of 42.5%**.

With respect to the batch size, I found 32 to be a good value. A bigger value would result in the training running faster, but it usually affects the performance of the machine. This probably wouldn't be a problem in my case, since I was using an Amazon AWS instance, but since it was running fast enough I didn't tweak this parameter.

* The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.

All the layers have been explained in the points above.

* The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

During the encoding stage, the *main* features (some information is lost in the process mainly due to the max-pooling) of the image are identified. The deeper the layer, the smaller the features. In the decoding layers, the previously detected features are identified within the image.

* The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.

We could reuse this architecture to follow another object but we would need to train it with the corresponding dataset and tune the hyperparameters. If the network doesn't fit our model we would need to change it, but there wouldn't be big differences as it is pretty standard.

## B. Model
* The model is submitted in the correct format.

[config_model_weights](../data/weights/config_model_weights)
[model_weights](../data/weights/model_weights)

* The neural network must achieve a minimum level of accuracy for the network implemented.

I've obtanid an IoU of 42.5%

### Further improvements
1. Get more data.
2. Test changes in the architecture, for instance deeper networks, dropout, activation functions...
