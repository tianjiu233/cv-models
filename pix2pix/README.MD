# This is a implementation of the algorithm pix2pix.(unstable)

### The details of the implementation.
The networch architecture used here is the same with the one in the [paper](https://arxiv.org/abs/1611.07004).
(There maybe be a little modification. The final encoder is without batchnorm.)
The algorithm is Adam with a learning rate of 2e-4 for both discriminator and generator.

### Notice:
I do not get the result as well as the [project](https://phillipi.github.io/pix2pix/) presents.



### Results:
All the pictures are obtained from train datasets. For some reseaon, when I run the model on the test dataset, the result are not so well.
