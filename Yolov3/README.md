This is a repo for yolov3 with pytorch

The codes are heavily borrow from [Erik](https://github.com/eriklindernoren/PyTorch-YOLOv3) and I make some changes, like removing the .cfg file.

The codes can be run in main.py.


PS: For the limitation of hardware, the dataset used here is GDUT-HWD, which is a hard-hat(or helmet) dataset released in ["Automatic detection of hardhats worn by construction personnel: A deep learning approach and benchmark dataset"](https://www.researchgate.net/publication/336184243_Automatic_detection_of_hardhats_worn_by_construction_personnel_A_deep_learning_approach_and_benchmark_dataset). The results we get is not the best one, but we do not train the model anymore.

The result is below, when the model is trained after about 64 epochs.:

| category |  ap   |
| :------: | :---: |
|   blue   | 0.684 |
|  white   | 0.658 |
|  yellow  | 0.733 |
|   red    | 0.618 |
|   none   | 0.603 |
|   MAP    | 0.659 |


The below are some result examples.


![example_1](https://github.com/tianjiu233/detection-models/blob/master/Yolov3/result/img_idx_1.png)
![example_2](https://github.com/tianjiu233/detection-models/blob/master/Yolov3/result/img_idx_4.png)
![example_3](https://github.com/tianjiu233/detection-models/blob/master/Yolov3/result/img_idx_5.png)
![example_4](https://github.com/tianjiu233/detection-models/blob/master/Yolov3/result/img_idx_8.png)
![example_5](https://github.com/tianjiu233/detection-models/blob/master/Yolov3/result/img_idx_17.png)
![example_6](https://github.com/tianjiu233/detection-models/blob/master/Yolov3/result/img_idx_23.png)
![example_7](https://github.com/tianjiu233/detection-models/blob/master/Yolov3/result/img_idx_21.png)
![example_8](https://github.com/tianjiu233/detection-models/blob/master/Yolov3/result/img_idx_14.png)


If you have any question, please to contact me.

h.j.
