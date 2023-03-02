# AlexNet 
This is an implementation of AlexNet, which is published in 2012 in the paper "ImageNet Classification
with Deep Convolutional Neural Networks" by Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton. 

## Points 
Points </br>
	1. Large, deep convolutional neural network to classify 1.2 million images </br>
        5 conv2d + 3 fc + softmax  --> 60 million parameters and 650,000 neurons. </br>
	2. some technologies </br>
		1. Overlap pooling - reduce overfit </br> 
		2. drop out  - to reduce overfit </br>
		3. Training on Multiple GPU </br>
		4. use ReLU activation function - not sigmoid activation function  - learning faster than $tanh$ function </br>
		5. Data Augmentation   - enlarge dataset to reduce overfit </br>
		6. Local Response Normalization - LRN -    </br>
	3. A large, deep convolutional neural network is capable of achieving record-breaking results on a highly challenging dataset using purely supervised learning. </br>
	4. The authors want to use very large and deep convolutional nets on video sequence where the temporal structure provides very helpful information that is missing or far less oobvious in static images. </br> 


