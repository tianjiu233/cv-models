# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 21:50:34 2018

@author: huijian
"""
import torch.nn as nn
import torch.functional as F

class myCELoss(nn.Module):
    def __init__(self,size_average=True):
        super(myCELoss,self).__init__()
        self.size_average = size_average
    def forward(self,predict,target,weight=None):
        """
        Args:
            predict:(n,c,h,w)
            target:(n,1,h,w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert predict.size(0) == target.size(0)
        assert predict.size(2) == target.size(2)
        assert predict.size(3) == target.size(3)

        predict = predict.permute(0,2,3,1).contiguous()
        predict = predict.view(-1,predict.size()[1])
        target = target.view(-1)
        loss=F.cross_entropy(predict,target,weight=weight,size_average=self.size_average)
        return loss

# black is removed from the COLOR_DICT
COLOR_DICT = (
"aliceblue",
"antiquewhite",
"aqua",
"aquamarine",
"azure",
"beige",
"bisque",
"blanchedalmond",
"blue",
"blueviolet",
"brown",
"burlywood",
"cadetblue",
"chartreuse",
"chocolate",
"coral",
"cornflowerblue",
"cornsilk",
"crimson",
"cyan",
"darkblue",
"darkcyan",
"darkgoldenrod",
"darkgray",
"darkgreen",
"darkgrey",
"darkkhaki",
"darkmagenta",
"darkolivegreen",
"darkorange",
"darkorchid",
"darkred",
"darksalmon",
"darkseagreen",
"darkslateblue",
"darkslategray",
"darkslategrey",
"darkturquoise",
"darkviolet",
"deeppink",
"deepskyblue",
"dimgray",
"dimgrey",
"dodgerblue",
"firebrick",
"floralwhite",
"forestgreen",
"fuchsia",
"gainsboro",
"ghostwhite",
"gold",
"goldenrod",
"gray",
"green",
"greenyellow",
"grey",
"honeydew",
"hotpink",
"indianred",
"indigo",
"ivory",
"khaki",
"lavender",
"lavenderblush",
"lawngreen",
"lemonchiffon",
"lightblue",
"lightcoral",
"lightcyan",
"lightgoldenrodyellow",
"lightgray",
"lightgreen",
"lightgrey",
"lightpink",
"lightsalmon",
"lightseagreen",
"lightskyblue",
"lightslategray",
"lightslategrey",
"lightsteelblue",
"lightyellow",
"lime",
"limegreen",
"linen",
"magenta",
"maroon",
"mediumaquamarine",
"mediumblue",
"mediumorchid",
"mediumpurple",
"mediumseagreen",
"mediumslateblue",
"mediumspringgreen",
"mediumturquoise",
"mediumvioletred",
"midnightblue",
"mintcream",
"mistyrose",
"moccasin",
"navajowhite",
"navy",
"oldlace",
"olive",
"olivedrab",
"orange",
"orangered",
"orchid",
"palegoldenrod",
"palegreen",
"palevioletred",
"papayawhip",
"peachpuff",
"peru",
"pink",
"plum",
"powderblue",
"purple",
"red",
"rosybrown",
"royalblue",
"saddlebrown",
"salmon",
"sandybrown",
"seagreen",
"seashell",
"sienna",
"silver",
"skyblue",
"slateblue",
"slategray",
"slategrey",
"snow",
"springgreen",
"steelblue",
"tan",
"teal",
"thistle",
"tomato",
"turquoise",
"violet",
"wheat",
"white",
"whitesmoke",
"yellow",
"yellowgreen")