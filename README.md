# Rotation-equivariant convolutional neural network ensembles in image processing
<img src="https://github.com/LouiseHash/Rotation_Equivariant_CNN_Ensembles/blob/master/figures/Fig1.PNG" width="600">

## Published in UbiComp/ISWC '19 Adjunct 2019

#### Abstract:
For the present engineering of neural networks, rotation invariant is hard to be obtained. Rotation symmetry is an important characteristic in our physical world. In image recognition, using rotated images would largely decrease the performance of neural networks. This situation seriously hindered the application of neural networks in the real-world, such as human tracking, self-driving cars, and intelligent surveillance. In this paper, we would like to present a rotation-equivariant design of convolutional neural network ensembles to counteract the problem of rotated image processing task. This convolutional neural network ensembles combine multiple convolutional neural networks trained by different ranges of rotation angles respectively. In our proposed theory, the model lowers the training difficulty by learning with smaller separations of random rotation angles instead of a huge one. Experiments are reported in this paper. The convolutional neural network ensembles could reach 96.35% on rotated MNIST datasets, 84.9% on rotated Fashion-MNIST datasets, and 91.35% on rotated KMNIST datasets. These results are comparable to current state-of-the-art performance.

#### Architecture: 
<img src="https://github.com/LouiseHash/Rotation_Equivariant_CNN_Ensembles/blob/master/figures/Fig2.PNG" width="700">

The image above could help to understand the architecture. We are combining results of convolutional neural networks, trained by different sets of angles, into an encoded set. Then, we use another fully-connected neural network to produce a prediction. The encoded set could be considered as a rotation invariant in our case. 

#### Experimental setup and results

Our experiment is built on ResNet-18. We use an ensemble of ResNet-18 to solve the problem. The performances is reported as the table below. We could see its ability for predicting harder datasets, such as KMNIST. 

<img src="https://github.com/LouiseHash/Rotation_Equivariant_CNN_Ensembles/blob/master/figures/Fig3.PNG" width="700">
