# An implementation of neural style transfer.
All the codes are similar to the tutorial [pyotrch.org](http://pytorch.org/) 

I just do some little modifications.

When reading the tutorial, I was really confused about using so many ```.clone() ``` and ``` .detach()```.
And I write my version and tried to reduce some of them.
#### ```.detach()```
I found that only two ```.detach``` were necessary for the function ```.MSELoss(output, target)``` requires that 
``` target ``` should be ``` requires_grad=False ```. Hence, you can use ```.detach()``` or do as following:

```
# self.target = target.detach()*weight
# In the original version, the api .detach() was used to make
# sure that requires_grad == False, Because it is required by criterion function.
self.target = Variable(target.data.clone(),requires_grad=False)*weight
```

#### ```.clone()```
This method involves the conception ["in-place operation"](http://deeplearning.net/software/theano_versions/dev/extending/inplace.html).
And you may find more about this concept.
With pytorch, you are not allowed to do in-place operation on the Variables that you want to do back-propagation.
And I find only ```.clone()``` in the ``` forward() ``` of ```StyleLoss``` is required.
(Sorry, I actually do not know why it is required here and I find it may be caused by GramMatrix. I add ```.clone()``` to the position ```gram()``` instead of ```self.output=input.clone()``` )

### Experiment
#### Input
Style Picture
[]!(https://github.com/huijianpzh/segmentation-models/blob/master/simple_style_transfer/data/dancing.jpg)
Content Picture
[]!(https://github.com/huijianpzh/segmentation-models/blob/master/simple_style_transfer/data/picasso.jpg)
#### Result
[]!()

h.j.
