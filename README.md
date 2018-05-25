# Sign language image recognition and example of overfitting
- We use a ResNet50 model to recognize how many fingers someone is holding up
- We apply the model to multiple sets of images, taken from different sources:
-
- ResNet50 implementation can be found [here](./resnet_utils.py)
- We import pre-trained weights from the Week 2 assignment of [Convolution Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks/)
- We apply the model to both the same show that the pre-trained weights are over-fitted
- We re-train the final layer of the model using a complete