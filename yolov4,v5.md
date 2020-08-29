参考代码：https://github.com/Tianxiaomo/pytorch-YOLOv4

# YOLOv4数据预处理

> 数据集准备

以VOC数据集的格式存放，使用精灵标注软件对图片进行标注，导出VOC格式文件，获得xml文件

根据xml文件生成用于训练的labels文件夹，文件夹中保存的文件是以图片名称命名的txt文件，内容是bbox的数值和目标类别，格式如下：

```
x1,y1,x2,y2,type
x1,y1,x2,y2,type
...
```

- x1,y1 : 左上角坐标
- x2,y2 : 右下角坐标
- type : 目标类别

之后，根据train和val数据比例自己打分，生成最终的label

> 配置文件修改

修改data文件夹下的`voc.names`，将其中的类别改为自己的训练集类别

> 训练参数修改

修改cfg文件夹下的`yolov4.cfg`中的相关参数：

- `[net]`下的训练参数，如batch，width，height等
- 最后一层`[convolutional]`的filters个数（filters=(5+类别数）*3）
- `[yolo]`下的classes类别数



# YOLOv4数据增强

代码详见：`./dataset.py`

YOLO V4使用了上图中多种数据增强技术的组合，对于单一图片，除了经典的几何畸变与光照畸变外，还使用了图像遮挡(Random Erase，Cutout，Hide and Seek，Grid Mask ，MixUp)技术，对于多图组合，作者混合使用了CutMix与Mosaic技术。除此之外，作者还使用了Self-Adversarial Training (SAT)来进行数据增强

**图像遮挡：**

- Random Erase：用随机值或训练集的平均像素值替换图像的区域
- Cutout：仅对 CNN 第一层的输入使用剪切方块Mask
- Hide and Seek：将图像分割成一个由 SxS 图像补丁组成的网格，根据概率设置随机隐藏一些补丁，从而让模型学习整个对象的样子，而不是单独一块，比如不单独依赖动物的脸做识别
- Grid Mask：将图像的区域隐藏在网格中，作用也是为了让模型学习对象的整个组成部分。
- MixUp：图像对及其标签的凸面叠加

**多图组合：**

- CutMix：对两张图片进行拼接变为一张新的图片，然后将拼接好了的图片传入到神经网络中去学习
- Mosaic：参考了CutMix数据增强方式, 是CutMix数据增强方法的改进版。利用四张图片，对四张图片进行拼接，每一张图片都有其对应的框框，将四张图片拼接之后就获得一张新的图片，同时也获得这张图片对应的框框，然后将这样一张新的图片传入到神经网络当中去学习，相当于一下子传入四张图片进行学习了



# YOLOv4训练

代码详见：`./train.py`

主要遇到的问题：环境安装与配置

windows下安装pycocotools

- 需要Visual C++ 编译环境：安装Build Tools for Visual Studio 2015及以上



# YOLOv4损失函数

代码详见：`/tool/utils_iou.py`

YOLOv4中计算了以下4种损失：

- IOU损失：1与预测框A和真实框B之间交并比的差值

![](https://www.zhihu.com/equation?tex=L_%7BIOU%7D%3D1-IOU%28A%2CB%29)

- GIOU损失：在IOU损失的基础上加上一个惩罚项，目的是为了缓解IOU损失在检测框不重叠时出现的梯度问题

![](https://www.zhihu.com/equation?tex=L_%7BGIOU%7D%3D1-IOU%28A%2CB%29%2B%5Cleft%7C+C-A%5Ccup+B%5Cright%7C%2F%5Cleft%7C+C%5Cright%7C)

- DIOU损失：在IOU损失的基础上加了一个简单的惩罚项，用来**最小化**两个检测框中心点的**标准化距离，**这样可以加速损失的收敛过程

![](https://www.zhihu.com/equation?tex=L_%7BDIOU%7D%3D1-IOU%28A%2CB%29%2B%5Crho%5E%7B2%7D%28A_%7Bctr%7D%2CB_%7Bctr%7D%29%2Fc%5E%7B2%7D)

- CIOU损失：考虑到了三个**几何因素**，分别为（1）重叠面积（2）中心点距离（3）长宽比

![](https://www.zhihu.com/equation?tex=L_%7BCIOU%7D%3D1-IOU%28A%2CB%29%2B%5Crho%5E%7B2%7D%28A_%7Bctr%7D%2CB_%7Bctr%7D%29%2Fc%5E%7B2%7D%2B%5Calpha.v)

公式参数解析：

- A : 预测框 ；B：真实框；C是A和B的最小包围框
- ![[公式]](https://www.zhihu.com/equation?tex=A_%7Bctr%7D) : 预测框中心点坐标
-  ![[公式]](https://www.zhihu.com/equation?tex=B_%7Bctr%7D) ：真实框中心点坐标
- ![[公式]](https://www.zhihu.com/equation?tex=%5Crho%28.%29) 是**欧式距离**的计算
- c 为 A , B 最小包围框的对角线长度
-  ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 是一个正数， ![[公式]](https://www.zhihu.com/equation?tex=v) 用来测量长宽比的一致性；![[公式]](https://www.zhihu.com/equation?tex=%5Calpha.v+) 表示对长宽比的惩罚项，若真实框和预测框的宽高相似，那么 ![[公式]](https://www.zhihu.com/equation?tex=v) 为0，该惩罚项就不起作用了。这个惩罚项作用就是控制**预测框的宽高**能够尽可能**快速**地与**真实框的宽高**接近。



# YOLOv4后处理

代码详见：`./tool/utils.py`

YOLOv4中采用了DIoU-NMS非极值抑制方法进行后处理

NMS是目标检测中必备的后处理步骤，目的是用来去除重复框，留下最精确的bbox

在经典的NMS中，得分最高的检测框和其它检测框逐一算出一个对应的IOU值，并将该值超过NMS阈值的框全部过滤掉。可以看出，在经典NMS算法中，IOU是**唯一**考量的因素。

但是在实际应用场景中，当两个不同物体挨得很近时，由于IOU值比较大，往往经过NMS处理后，只剩下一个检测框，这样导致漏检的错误情况发生。

基于此，==DIOU-NMS就不仅仅考虑IOU，还考虑两个框中心点之间的距离==。如果两个框之间IOU比较大，但是两个框的距离比较大时，可能会认为这是两个物体的框而不会被过滤掉。 其公式如下：

![](https://img-blog.csdnimg.cn/20200610110949759.png)



# YOLOv5

参考代码：

https://github.com/XiaoJiNu/yolov5-v1-chinese-comment

https://github.com/ultralytics/yolov5



## 网络结构对比

现阶段的目标检测器主要由4部分组成： **Input**、**Backbone**、**Neck**、**Head**。

![](https://img-blog.csdnimg.cn/20200514145553469.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU2MDQwMg==,size_16,color_FFFFFF,t_70#pic_center)

- **Input**：输入图像
- **Backbone**： 在不同图像细粒度上聚合并形成图像特征的卷积神经网络

- **Neck**：一系列混合和组合图像特征的网络层，并将图像特征传递到预测层

- **Head**： 对图像特征进行预测，生成边界框和并预测类别



> YOLOv4

![](https://img-blog.csdnimg.cn/20200806234945238.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25hbjM1NTY1NTYwMA==,size_16,color_FFFFFF,t_70#pic_center)



YOLOv4模型由以下部分组成：

- BackBone：**CSPDarknet53**（CSPNet[^1]+Darknet53，**作用**：**增强CNN的学习能力，能够在轻量化的同时保持准确性、降低计算瓶颈、降低内存成本**）
- Neck的附加模块：**SPP**[^2]（Spatial Pyramid Pooling ，空间金字塔池化，**作用**：**增加网络的感受野**）
- Neck的特征融合模块：**PANet**[^3]（作用：进行上采样操作）
- Head：**YOLOv3**



> YOLOv5

![](https://img-blog.csdnimg.cn/20200815154652781.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25hbjM1NTY1NTYwMA==,size_16,color_FFFFFF,t_70#pic_center)



YOLOv5模型由以下部分组成：

- BackBone：**Focus+Bottleneck(ResNet)+CSP**

  - Focus隔行采样拼接结构：YOLOv5默认3x640x640的输入，复制四份，然后通过切片操作将这个四个图片切成了四个3x320x320的切片，接下来使用concat从深度上连接这四个切片，输出为12x320x320，之后再通过卷积核数为32的卷积层，生成32x320x320的输出，最后经过batch_borm 和leaky_relu将结果输入到下一个卷积层

  ```python
  class Focus(nn.Module):
      def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
          super(Focus, self).__init__()
          self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
  
      def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
          return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
  ```

- Neck的附加模块：**SPP**[^2]

- Neck的特征融合模块：**PANet**[^3]



本节图片摘自：https://blog.csdn.net/nan355655600/article/details/107852288?utm_medium=distribute.pc_relevant.none-task-blog-baidulandingword-3&spm=1001.2101.3001.4242



[^1]: CSPNet：https://arxiv.org/pdf/1911.11929.pdf
[^2]:SPP：https://arxiv.org/pdf/1406.4729.pdf
[^3]:PANet： https://arxiv.org/pdf/1803.01534.pdf