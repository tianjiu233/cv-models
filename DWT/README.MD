# It is an implementation of DWT(Deep Watershed Transform).(not finished!)

The [Link](https://github.com/min2209/dwt) is the implementation by the author of the paper.

I will try to rewrite in pytorch and used in the RS images.
There may be some differences, but I am going to explain as clearly as I could.

### DN(Direction Network)
#### the directional unit vector
[Link](https://github.com/min2209/dwt/blob/master/matlab/generate_GT_cityscapes_unified.m) is about how to compute the ground truth of the univer vector map. 
I rewrite it using python.

## References:
* [Deep Watershed Transform for Instance Segmentation](https://arxiv.org/pdf/1611.08303.pdf)
