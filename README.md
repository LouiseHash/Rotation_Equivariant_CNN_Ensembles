# Rotation-equivariant convolutional neural network ensembles in image processing
<img src="https://github.com/LouiseHash/Rotation_Equivariant_CNN_Ensembles/blob/master/figures/Fig1.PNG" width="600">
In our real life, image might largely rotated. Take self-driving car as an example, as we can see in the figure above, the road sign could be rotated. If we use traditional convolutional neural networks, we might result in a low training accuracy. This work aims to solve this problem. In fact, there were a series of researches, using better filters and activation functions, performed rigorous analysis to this problem with promising results. If you are looking for solving this problem thoroughly in a rigorous way, this repo might not be a good place for you. 

However, if you are considering this problem more from a practical perspective, the architecture that we propose could be a good choice for you. The code is not hard to understand and implement. If you know how to train a convolutional neural network, it will be definitely easy for you to reimplement this algorithm. This algorithm simply trains several convolutional neural networks in different angles and combine them. This architecture is also not difficult to debug, and remains certain interoperability. We used ResNet in our work, but it could also be replaced by any convolutional neural networks such as VGG, U-Net, and so on. 

## [Published in UbiComp/ISWC '19 Adjunct 2019](https://dl.acm.org/ft_gateway.cfm?id=3349330&type=pdf)
[Recorded link in CPD-19.](https://www.youtube.com/watch?v=onOCPi-Fao4)


#### Abstract:
For the present engineering of neural networks, rotation invariant is hard to be obtained. Rotation symmetry is an important characteristic in our physical world. In image recognition, using rotated images would largely decrease the performance of neural networks. This situation seriously hindered the application of neural networks in the real-world, such as human tracking, self-driving cars, and intelligent surveillance. In this paper, we would like to present a rotation-equivariant design of convolutional neural network ensembles to counteract the problem of rotated image processing task. This convolutional neural network ensembles combine multiple convolutional neural networks trained by different ranges of rotation angles respectively. In our proposed theory, the model lowers the training difficulty by learning with smaller separations of random rotation angles instead of a huge one. Experiments are reported in this paper. The convolutional neural network ensembles could reach 96.35% on rotated MNIST datasets, 84.9% on rotated Fashion-MNIST datasets, and 91.35% on rotated KMNIST datasets. These results are comparable to current state-of-the-art performance.

#### Architecture: 
<img src="https://github.com/LouiseHash/Rotation_Equivariant_CNN_Ensembles/blob/master/figures/Fig2.PNG" width="700">

The image above could help to understand the architecture. We are combining results of convolutional neural networks, trained by different sets of angles, into an encoded set. Then, we use another fully-connected neural network to produce a prediction. The encoded set could be considered as a rotation invariant in our case. 

#### Experimental setup and results

Our experiment is built on ResNet-18. We use an ensemble of ResNet-18 to solve the problem. The performances is reported as the table below. We could see its ability for predicting harder datasets, such as KMNIST. 

<img src="https://github.com/LouiseHash/Rotation_Equivariant_CNN_Ensembles/blob/master/figures/Fig3.PNG" width="700">

#### Tricks in training
The only trick that we are using here is the overlapping of the training angles. Suppose we have 8 convolutional neural networks in this ensemble. As you could check in the jupyter notebook, there is a huge angle of overlapping for these ensemble members. It is not hard to explain since the neural networks will predict a misleading result for images with different angles, if the network is completely not trained by this angle. This will be hard to produce a stable training result. Instead, adding certain overlapping for each ensemble members could make this process more stable. 
#### We recommend you to run this code on Google's colab for simplicity. Future version of this code will be available for the next step. 
