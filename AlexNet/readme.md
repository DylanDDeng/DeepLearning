# AlexNet 
This is an implementation of AlexNet, which is published in 2012 in the paper "ImageNet Classification
with Deep Convolutional Neural Networks" by Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton. 

## Points

1. Large, deep convolutional neural network to classify 1.2 million images 
	1. 5 conv2d + 3 fc + softmax  --> 60 million parameters and 650,000 neurons. 
2. some technologies 
    1. Overlap pooling - reduce overfit 
    2. drop out  - to reduce overfit  
    3. Training on Multiple GPU 
    4. use ReLU activation function - not sigmoid activation function  - learning faster than tanh function  
    5. Data Augmentation   - enlarge dataset to reduce overfit 
    6. Local Response Normalization - LRN -    
3. A large, deep convolutional neural network is capable of achieving record-breaking results on a highly challenging dataset using purely supervised learning. 
4. The authors want to use very large and deep convolutional nets on video sequence where the temporal structure provides very helpful information that is missing or far less obvious in static images. 

## Files
```model.py``` is the AlexNet model built based on **Pytorch**   
```train.py``` is the script trained the model.  
```utils.py``` is the script to process our image data 
```my_dataset.py``` is the script to define the dataset to load 


### Reference 
I refer the following link about the code. 
[URL](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test2_alexnet)

