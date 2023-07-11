# Multi-layer-neural-networks

## Neural Networks for Image Memorization

This repository contains a multi-layer neural network implementation for image memorization, a unique approach of "teaching" a network to reconstruct an image from pixel coordinates. The work here is based on the idea that inputs to the multi-layer network are 2-dimensional (x,y) coordinates and the outputs of the network are the 3-dimensional RGB values.

## Table of Contents

- [Description](#Description)
- [Implementation](#Implementation)
- [Extra Credit](#Extra-Credit)
- [Acknowledgements](#Acknowledgements)

## Description

The aim of this project is to train a neural network to generate the pixel color values given the pixel's coordinates. This is achieved by using the pixel's (x, y) coordinate as the input, and the corresponding RGB value as the output during training. The network, after being trained, should be able to reconstruct the image by simply feeding it with the coordinates of the original image.

## Implementation

A four-layer neural network is implemented from scratch, and the training is conducted using both SGD and Adam optimizers. The forward and backward propagation algorithms have been implemented to train the model and update the weights, respectively. The network minimizes the Mean Squared Error (MSE) loss between the original and the reconstructed images.

The input features have been transformed using different forms of input feature mapping as outlined in this paper(https://bmild.github.io/fourfeat), which was found to improve the final image reconstruction.

The code, comments, and explanations for the same are provided in the Jupyter Notebook, `neural_net.py`.

## Extra Credit

For extra credit, we have implemented and explored some advanced applications and extensions:
- We designed a deeper neural network and analysed its results, computational requirements, and training behavior.
- We experimented with alternative losses such as L1 loss, regularization techniques, and normalization techniques covered in class, and noted the difference they made in the reconstruction.
  
## Acknowledgements

This project has been done as a part of a course assignment. We would like to thank the course instructors and TAs for their guidance and support throughout the project.
Please note that this project is meant for educational purposes and should be used responsibly.

---

Please refer to the `neural_net.py` notebook for more details on the project implementation, results, and discussions. If you have any questions or suggestions, feel free to open an issue.
