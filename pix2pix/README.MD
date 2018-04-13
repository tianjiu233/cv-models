# This is a implementation of the algorithm pix2pix.(unstable)

### The details of the implementation.
The networch architecture used here is the same with the one in the [paper](https://arxiv.org/abs/1611.07004).
(There maybe be a little modification. The final encoder is without batchnorm.)
The algorithm is Adam with a learning rate of 2e-4 for both discriminator and generator.

### Notice:
I do not get the result as well as the [project](https://phillipi.github.io/pix2pix/) presents.
I find some may write the codes about rgb2lab and lr_scheduler to adjust the learning rate. I did not do that.


### Results:
All the pictures are obtained from train datasets. For some reseaon, when I run the model on the test dataset, the result are not so well.

#### sample_1 (top:label, bottom:result)
![](https://github.com/huijianpzh/segmentation-models/blob/master/pix2pix/result/label_sample_1.jpg)
![](https://github.com/huijianpzh/segmentation-models/blob/master/pix2pix/result/result_sample_1.jpg)
#### sample_2
![](https://github.com/huijianpzh/segmentation-models/blob/master/pix2pix/result/label_sample_2.jpg)
![](https://github.com/huijianpzh/segmentation-models/blob/master/pix2pix/result/result_sample_2.jpg)
#### sample_3
![](https://github.com/huijianpzh/segmentation-models/blob/master/pix2pix/result/label_sample_3.jpg)
![](https://github.com/huijianpzh/segmentation-models/blob/master/pix2pix/result/result_sample_3.jpg)
#### sample_4
![](https://github.com/huijianpzh/segmentation-models/blob/master/pix2pix/result/label_sample_4.jpg)
![](https://github.com/huijianpzh/segmentation-models/blob/master/pix2pix/result/result_sample_4.jpg)

### To be continued:
I will continue to modify the code.
Hopefully, it will get a better result.

### References:
* [Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)
* [pix2pix(tensorflow version)](https://affinelayer.com/pixsrv/)
* [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [pix2pixBEGAN.pytorch](https://github.com/taey16/pix2pixBEGAN.pytorch)
