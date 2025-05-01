# fashion-mnist-classifier

## Data available
The data used in this project is the Fashion MNIST dataset, which includes 70,000 images of 10 classes of clothing images, each 28*28 pixels, with only 1 channel in the image. 
The data is usually split into 60,000 training images and 10,000 testing images. 

## Scope of the project
In this project, I plan to test 3 neural network models against this data and evaluate their accuracy using the same training and test data.

### 2 layer Fully connected neural network
This model is the most simplistic version of a neural network imaginable. It uses 1 hidden layer, and overall 2 fully connected layers.
The layers of the model are in this order:
- nn.Flatten(): to flatten the 28 by 28 image to a vector of length 784 (28*28)
- nn.LazyLinear(): a layer with an output of dimension 20 
- nn.ReLU(): as an activation of the hidden layer
- nn.Linear(): as a fully connected layer to use the 20 dimensions and get scores for the 10 classes

This model uses a crossEntropy loss function, and an SGD optimizer with a learning rate of 0.05 and a momentum of 0.9.
The performance of this model is 86.99% accuracy for the training data and 84.05% accuracy for the test data.

### 5 layer Convolutional Neural Network
This model is a more intelligent step towards a better image classifier, with 3 convolutional layers and 2 fully connected layers.
The layers of this model are created using 4 blocks. 
3 of the blocks are based on the same structure, which is as follows:
- nn.LazyConv2d(): a convolutional layer with custom padding and number of output channels, with a fixed stride of 1
- nn.ReLU(): as an activation function after the convolutional layer
- nn.MaxPool2d(): to pool the output of the convolutional layer 
The last block of the model is a 2 layer fully connected block, which is the same structure as the previous model, with the difference that the hidden layer has a dimension of 128 instead of the 20 in the previous model.
The layers of this model are in this order:
- a conv2d block with padding of 3 and output channels of 64 that turns the 1 * 28 * 28 shaped image into 64 * 16 * 16
- a conv2d block with padding of 1 and output channels of 64 that turns the 64 * 16 * 16 shaped image into 64 * 8 * 8
- a conv2d block with padding of 1 and output channels of 128 that turns the 64 * 16 * 16 shaped image into 128 * 4 * 4
- a final fully connected block that first flattens the image from 128 * 4 * 4 to a vector of size 2,048, then 128, and then 10 as the output class scores

This model also uses a crossEntropy loss function, but instead uses an Adam optimizer with a learning rate of 0.001.
The performance of this model is 98.69% accuracy for the training data and 91.99% accuracy for the test data.

