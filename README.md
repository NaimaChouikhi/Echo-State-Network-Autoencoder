# Echo-State-Network-Autoencoder
The presented code consists of an Echo State Network Recurrent Autoencoder ESN-RAE. In fact, sometimes the original data representations may not be the most expressive data distribution for a targeted task. The idea behind using AEs is to make the inputs equal to the outputs within a neural network then take the activations of the hidden layer as new data representation. ESN is used in this code as RAE to extract new data different from the original ones. Once the new data are extracted, they are squirted into an SVM classifier in order to get the classification accuracy.

*All details found in our paper http://arxiv.org/abs/1804.08996 

*Getting started

Run the script with Matlab: ESN-AE.m
