## Instance Segmentation using discriminative loss

* This is an implementation of the paper ["Semantic Instance Segmentation with a Discriminative Loss Function"](https://arxiv.org/abs/1708.02551).

* The codes are heavily borrowed from the [repository](https://github.com/Wizaron/instance-segmentation-pytorch) <br>
There are 2 ```.py``` about the discriminative loss. Both of them can work well. The ``` new_discriminative.py ``` is for the situation where sometimes the sample have no targets. <br>
The codes work on cpu, however, the enviroment with gpu are not tested for the limatation.(The codes for are included.)<br>
If you have any question, please contact me.

## Enviroment: 
* pytorch 0.4.0
* Win 10 / Ubuntu 16.04

## Result:
![](https://github.com/huijianpzh/segmentation-models/blob/master/instance/result_demo.png)

h.j.

