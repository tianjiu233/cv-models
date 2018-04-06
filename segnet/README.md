# An implementation of SegNet.

For the limitation of computational source, I changed the number of the filters in the network structure.

Maybe for that the filters have been reduced, it does not perform as well as U-Net. (Accuracy:90.14% Precision:78.27% Recall:79.08%)
(PS: You can build the SegNet model as decribed in the original paper, when you set the parameter ```filter``` as ```[64,128,256,512,512]```. And in this experiment, we use ```segnet = SegNet(input_dim=3, output_dim=1, features=[64,96,128,256,256])```)

The dataset used here is the same as the one in [U-Net](https://github.com/huijianpzh/segmentation-models/edit/master/segmentation_unet/README.md).

## Result:
![](https://github.com/huijianpzh/segmentation-models/blob/master/segnet/result.png)

## References:
* [Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding](https://arxiv.org/pdf/1511.02680.pdf)
