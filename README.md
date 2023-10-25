# 基于 YOLOv8 和 Intel 套件的城市景观识别

---

@[TOC](文章目录)

---

## 一、问题陈述

&ensp;&ensp;目前，安全导航问题作为自动驾驶车辆的一个研究重点，其要求分析模型能够快速地识别周围的复杂道路环境，并精确地检测出重点目标，强调“实时性”与“精确性”。借此，我们希望能够利用计算机视觉技术以及 IntelAI 分析工具套件为自动驾驶车辆创建一个实时的对象检测模型，实现其高精确性与低延迟性。<br>&ensp;&ensp;而在具体分析实际城市景观图像后，发现大多场景所描述的均为非结构化环境，其具有复杂的环境背景，难以识别的目标边界，以及多样化的遮蔽现象。这些因素对目标检测的准确度提出了极高的要求，将极大地影响整个模型的训练方法与最终得出的识别质量。同时，在此基础上还需考虑如何尽可能地提高检测速度，此即项目开发过程中将着重处理的问题。<br>&ensp;&ensp;由此，我们将比对优劣以挑选出一种适宜的目标检测算法用以创建模型，同时寻找一个既能够增强目标边缘检测准确度，又能够在一定程度上提高检测速度的方法，将其设计为一个图像预先处理的步骤加入到我们的训练模型之中，以解决上述复杂场景的识别问题。

---

## 二、项目简介

&ensp;&ensp;此次项目，我们将基于 YOLOv8 算法进行图像识别模型的创建，在此基础上使用 Intel 套件进行进一步的处理，同时利用图像的高频分量来进一步提高目标边缘（轮廓）的确定速度与识别精准度，以使得项目模型达到更好的效果与更高的性能。

---

## 三、数据集介绍

&ensp;&ensp;我们使用了 cityscapes 数据集[^1] ，该数据集包含超过 20000 帧街道场景中记录的各种立体视频序列，除了 20 000 帧弱注释帧外，还具有 5 000 帧的高质量的像素级注释帧。<br>&ensp;&ensp;此外，该数据集还具有相当的多样性，取样来自五十个不同的城市，场景包括不同的季节，不同的气象条件和一天中的不同时间。数据集制作者对此的描述是：

> supporting research that aims to exploit large volumes of (weakly) annotated data, e.g. for training deep neural networks.
> 支持旨在利用大量(弱)注释数据的研究，例如用于训练深度神经网络。

[^1]: https://www.cityscapes-dataset.com

---

## 四、数据集预处理

&ensp;&ensp;使用`torchvision.datasets.Cityscapes`类加载 cityscapes 数据集的训练集、测试集和验证集。下面仅展示训练集的加载

```python
dataset_train = torchvision.datasets.Cityscapes(
        root=origin_root,
        split='train',
        mode='fine',
        target_type='polygon'
    )
```

&ensp;&ensp;需要注意的是，cityscapes 数据集使用多边形进行标注，而 YOLOv8 使用的是矩形标注，我们要对标注方式进行转换

```python
h, w, objs = label['imgHeight'], label['imgWidth'], label['objects']
txt = ''
for obj in objs:
    # get correct label
    if small_label.get(obj['label'], None) is not None:
        label = small_label[obj['label']]
    else:
        label = obj['label']

    if label2idx.get(label, None) is None:
        continue
    else:
        # get the bbox using the polygon
        x_min = w
        y_min = h
        x_max = 0
        y_max = 0
        for point in obj['polygon']:
            x_min = min(x_min, point[0])
            x_max = max(x_max, point[0])
            y_min = min(y_min, point[1])
            y_max = max(y_max, point[1])

        x = (0.0+x_min+x_max)/(w)
        y = (0.0+y_min+y_max)/(h)
        h_ = (0.0+y_max-y_min)/(h)
        w_ = (0.0+x_max-x_min)/(w)
        txt += f'{label2idx.get(label)} {x} {y} {w_} {h_}\n'
```

&ensp;&ensp;简单来说，我们对多边形取外接矩形，用矩形代替原本的多边形 。YOLOv8 中使用矩形的中心点坐标:`(x,y)`，长:`w`，宽:`h`，来表示矩形的轮廓位置，而我们使用的外接矩形是用四个顶点来表示，所以需要再次进行转换。  
&ensp;&ensp;此外，我们会使用`C_ADD`类提取的物体轮廓特征，将它作为一个新的 channel，与原先的 RGB 一同传入模型，对物体检测的过程进行辅助。所以在修改标注框的过程中，我们在前 100 个 epoch 中稍作修改，让标注框变小，方便模型学习物体的内部纹理，具体如下：

```python
#Reduce the length and width of the box by half
x = (0.0+x_min+x_max)/(w)
y = (0.0+y_min+y_max)/(h)
h_ = (0.0+y_max-y_min)/(h*2)
w_ = (0.0+x_max-x_min)/(w*2)
```

## 五、调用 YOLOv8 和 Intel 套件

### 1.YOLOv8 简介

YOLOv8 的官方架构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/1bef36ad41f0464f90973e1fd9a83a53.jpeg)

1. 提供了模型 SOTA 模型，其中包括了目标检测网络和实例分割模型 YOLACT，YOLACT 架构如下，蓝色/黄色表示原型中的低/高值，灰色节点表示未训练的函数，本例中的 k = 4，图片摘自[论文原文](https://arxiv.org/pdf/1904.02689.pdf)
   ![YOLACT](https://img-blog.csdnimg.cn/0611f5dcf8994c2cb9226b539b7bd407.png)

2. YOLOv8 适用于 P5 640 和 P6 1280 分辨率。与 YOLOv5 类似，设置了不同尺度的模型（N/S/M/L/X），对于本次的数据集规模，我们选用的尺度为 M 的模型。

```python
# Parameters
nc: 80  # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024]  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768]   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512]   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512]   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
```

2. YOLOv8 模型将 YOLOv5 的 C3 结构替换为了更丰富梯度流的 C2f 结构(如下)，并对不同尺度的模型进行了不同的通道数调整，而不再是一套参数适用于所有模型，从而显著提升了性能。
   ![C2f](https://img-blog.csdnimg.cn/671ddeaaaf6e4bd98bd15265d0a01775.jpeg)
3. 与 YOLOv5 相比，模型的 Head 部分（如下）经历了较大的改动，采用了解耦头结构，将分类和检测头分离，并从 Anchor-Based 转向了 Anchor-Free。
   ![YOLOv8 Head](https://img-blog.csdnimg.cn/ad5d7bd2b62c496e8d6c11c92236cad4.png)

4. 在损失计算方面，采用了 TaskAlignedAssigner 正样本分配策略，并引入了 Distribution Focal Loss。

5. 训练过程中引入了 YOLOX 中的最后 10 个 epoch 关闭 Mosiac 增强的操作，以有效提高精度，对比如下。
   ![对比](https://img-blog.csdnimg.cn/538022348a7546bf9ee185c178bdf548.jpeg)

### 2.调用过程

#### 调用 YOLOv8 并进行修改

&ensp;&ensp;YOLOv8 实现了模型定义与训练的分离，模型定义在`yolov8.yaml`文件中，模型使用的 Settings 和 Hyperparameter 定义在`default.yaml`文件中，通过`tasks.py`文件的`parse_model`函数，读取`yolov8.yaml`文件并构建模型。我们可以修改`parse_model`函数，从而将自己对模型的修改嵌入原先的模型，例如，先在`nn.Module.block.py`中定义`C_ADD`模型，然后在`parse_model`函数中加入如下代码，即可将我们定义的 [C_ADD](#section1)模型加入其中

```python
if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
         BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3):
     #Initialization of different types of models
elif m is C_ADD:
   c2 = 4
```

#### 调用 Intel oneAPI 套件

&ensp;&ensp;模型的训练过程定义在`trainer.py`中，可以在其中调用Intel oneAPI 套件中`intel_extension_for_pytorch`的方法对模型进行压缩，我们主要使用了知识蒸馏（Knowledge Distillation），量化（Quantization）和剪枝（Pruning）对模型进行压缩。
&ensp;&ensp;剪枝就是通过去除网络中冗余的通道（channels），滤波器（filters）,神经元节点（ neurons）, 网络层（layers）以得到一个更轻量级的网络，同时不影响性能。
![Pruning](https://img-blog.csdnimg.cn/3dad98f0b658405096b5e23615dbfdfd.png)
&ensp;&ensp;知识蒸馏是通过构建一个轻量化的小模型，利用性能更好的大模型的监督信息，来训练这个小模型，以期达到更好的性能和精度。这个大模型我们称之为 teacher（教师模型），小模型我们称之为 Student（学生模型）。来自 Teacher 模型输出的监督信息称之为 knowledge(知识)，而 student 学习迁移来自 teacher 的监督信息的过程称之为 Distillation(蒸馏)。
![Distillation](https://img-blog.csdnimg.cn/b7d497c3d23640949dd8f9d4be9499c8.png)
&ensp;&ensp;模型量化即，将模型中的网络参数，由连续取值的高精度浮点数（如float32）转换为离散的低精度整数（如int8）的过程，同时保证模型的输入输出类型不变（依然是浮点数），从而在部分精度损失的前提下，达到减少模型尺寸大小、减少模型内存消耗及加快模型推理速度等目标，提升网络的推理性能。精度的变化并不是简单的强制类型转换，而是为不同精度数据之间建立一种数据映射关系，例如下图所示的FLOAT32与有符号INT8的映射过程：
![映射](https://img-blog.csdnimg.cn/4b0e447661c0489e857fd968157054cd.png)
&ensp;&ensp;调用Intel oneAPI的量化方法的代码如下：
```python
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig

recipes = {
    "smooth_quant": True,
    "smooth_quant_args": {
        "alpha": 0.5,
    },  # default value is 0.5
    "fast_bias_correction": False,
    "max_trial": 10000,
    "time_out": 1000
}

dataloader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=1,
    shuffle=False
)

conf = (
    PostTrainingQuantConfig(recipes=recipes)
)  # default approach is "auto", you can set "dynamic":PostTrainingQuantConfig(approach="dynamic")
q_model = quantization.fit(
    model=model,
    conf=conf,
    calib_dataloader=dataloader,
)
q_model.save("./output")
```
&ensp;&ensp;调用Intel oneAPI的知识蒸馏和剪枝方法的代码如下：

```python
def _setup_train(self, world_size):

        # Initialization of Model
       '''Omit them here'''
        # intel neural compressor set up
        self.DisCri = KnowledgeDistillationLossConfig()
        self.DisConf = DistillationConfig(self.model, self.DisCri)
        self.PrunConf = WeightPruningConfig(configs)
        self.Com_Manager = prepare_compression(model=self.model, confs=[self.DisConf, self.PrunConf])

        self.model.manager = self.Com_Manager

        # Other tasks
        '''Omit them here'''
```

&ensp;&ensp;同时，如前文所示，我们还需要利用 Intel 套件中的`self.Com_Manager`实现 hook 机制，当特定事件发生时，运行所有相关的回调函数,这里仅展示两个例子:
&ensp;&ensp;例如，每一个 epoch 开始时:

```python
# Other tasks
'''Omit them here'''
for epoch in range(self.start_epoch, self.epochs):
    # intel neural compressor hook
    self.Com_Manager.callbacks.on_epoch_begin(epoch)
    # Other tasks
    '''Omit them here'''
```

&ensp;&ensp;以及，训练结束时:

```python
# Other tasks
'''Omit them here'''
self.Com_Manager.on_train_end()
# Other tasks
'''Omit them here'''
```


## 六、创新点
<a id="section1"></a>
### 创新点 1

&ensp;&ensp;在训练之前，我们利用 Sobel 算子对图像进行处理，Sobel 算子如下图：

```python
class C_ADD(nn.Module):
    # this add the forth channel for img
    def __init__(self):
        super(C_ADD, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        self.conv.weight.data = torch.tensor(#Sobel算子
												[[[[-1.0, -1.0, 0],
                                                [1.0, 1.0, 0],
                                               [0.0, 0.0, 0.0]],
                                                [[-1.0, -1.0, 0],
                                                [1.0, 1.0, 0],
                                               [0.0, 0.0, 0.0]],
                                                [[-1.0, -1.0, 0],
                                                [1.0, 1.0, 0],
                                               [0.0, 0.0, 0.0]]]], dtype=torch.float32)

    def forward(self, x):
        with torch.no_grad():
            return torch.cat((x, torch.abs(self.conv(x))), dim=1)
```

&ensp;&ensp;在此处定义的这个类是将输入的图像在通道维度上扩展，通过对输入图像的每个通道应用相同的卷积核，得到一个具有额外通道的特征图，用于增强图像上的边缘特征。
&ensp;&ensp;这段代码的具体作用是设置一个卷积层的权重参数，将一个大小为 3\*3 的卷积核的权重设置为一个 4 维张量。每个卷积核的权重用一个 3\*3 的矩阵表示。
&ensp;&ensp;这个权重矩阵的作用是在卷积操作中对输入数据进行滤波。通过对输入数据的不同区域进行加权求和，卷积操作可以提取输入数据的不同特征。
&ensp;&ensp;而我们使用的 Sobel 算子由于在垂直方向上差异很大，所以会放大原来图片垂直方向上的差异，更加便于过滤出图片中的个物体的边缘轮廓。可以去除不必要的噪声、异常值或缺失值，有助于提高数据质量，减少对模型的负面影响，也加速了机器学习的训练过程，提高收敛速率。
&ensp;&ensp;最后用 cat 函数将输入张量 x 和卷积结果的绝对值在通道维度上进行拼接，得到一个具有四个通道的张量。

### 创新点 2

&ensp;&ensp;在之前的数据预处理中，曾提到过我们缩小了一部分标注框来突出物体内部纹理：

```python
#Reduce the length and width of the box by half
x = (0.0+x_min+x_max)/(w)
y = (0.0+y_min+y_max)/(h)
h_ = (0.0+y_max-y_min)/(h*2)
w_ = (0.0+x_max-x_min)/(w*2)
```

&ensp;&ensp;缩小标注框并突出物体内部纹理有以下好处：
1. 提高标注的准确性：缩小标注框可以更精确地捕捉物体的边界和形状，减少标注误差。突出物体内部纹理可以更好地理解物体的结构和特征，从而减少错误。
2. 提高标注的一致性：突出物体内部纹理可以提供更多的信息和细节，缩小标注框可以使在标注同一类物体时更加一致，减少标注结果的差异性。
3. 提高模型的性能：缩小标注框和突出物体内部纹理可以提供更多的训练样本和更丰富的特征，从而改善机器学习模型的性能。模型可以更好地学习物体的形状、纹理和结构，提高对物体的检测和分类能力。

&ensp;&ensp;总之，缩小标注框并突出物体内部纹理可以提高标注的准确性和一致性，同时改善机器学习模型的性能，加快收敛速度。这对于训练高质量的视觉模型和实现准确的目标检测任务非常重要。

---

## 七、结果展示
&ensp;&ensp;衡量预测结果的指标有两个，即：
1. 像素交并比 (Intersection over Union， loU ) 
2. 像素精度 (Pixel Accuracy， PA ) 
&ensp;&ensp;计算公式和代码实现如下：
![公式](https://img-blog.csdnimg.cn/4581dd334db64d0d9c4ea43d529ca67b.png)

```python
def evaluate(class_id, pred, target, padding=(0, (2048-1024)/2)):
    # pred is in the form of(x1, y1, x2, y2, conf, class_id)
    prediction = torch.full(fill_value=-1, size=(target.shape[-1], target.shape[-2]), device=target.device)
    prediction.fill_(-1)
    target = torch.where(target == class_id, class_id, -1)
    pred_ = pred.clone()
    pred_[:, (0, 2)] -= padding[0]
    pred_[:, (1, 3)] -= padding[1]
    for p in pred_:
        if p[-1] == class_id:
            prediction[int(p[1]+0.5):int(p[3]+0.5), int(p[0]+0.5): int(p[2]+0.5)] = class_id
    prediction = prediction.permute([1, 0])
    tp = torch.where((target == class_id)*(prediction ==class_id), 1, 0).sum()
    fp = torch.where((prediction == class_id) * (target == -1), 1, 0).sum()
    tn = torch.where((target == -1) * (prediction == -1), 1, 0).sum()
    fn = target.shape[-1] * target.shape[-2] - tp - fp - tn
    return tp, fp, tn, fn
```
&ensp;&ensp;结果如下：
| Label| IoU|PA|
| :----:| :----: | :----: |
|Person| 0.48552970388835365|.7279142150878907
|Bicycle | 0.08947700874469611|0.8056015195846558
|Car|0.00768193270175575|0.96415696144104
|Motorcycle|0.0 |0.9946085271835328
|Airplane|0.27010629767401745|0.6530777482986451
|Bus|IoU: 0.007199373473069394|0.9738247804641723
|Train|0.020684682738711425|0.9808716325759887
|Truck| 0.0|0.9997403831481934
|Boat| 0.020121417051213557|0.9508402137756348
|Traffic_light|0.004320607635808001|0.9960992450714111
|Fire_hydrant| 0.007547349887377524|0.9869889640808105
|Stop_sign|0.029209640843936544|0.9630387487411499
|Parking_meter|0.1216936621495841|0.8831227807998657
|Bench|0.0|0.9996036043167115
|Bird| 0.003194779316257256|0.9965126371383667
|Cat:|8.068672562841587e-05|0.9978963041305542
|Dog| 0.01680004317676792|0.98488524723052984
|Total|0.2708139603883966|0.9328696184158325

&ensp;&ensp;可以发现，机动车和各种路标的的IoU和PA较高，效果较好，而在公路上不常见的物体，如动物和人则效果较差。推理速度为33ms(硬件为3080Ti)

## 八、团队心得

**YOLOv8：**
&ensp;&ensp;首先，在选择算法的过程中，我们进一步了解到了 YOLOv8 的框架，同时通过其与 YOLOv5 的对比，了解到了其在结构与参数上的各项改动，认识到了这些改动所带来的优势之处和这些改动会如何优化训练模型的性能。在此基础之上进行合理修改，使其更加灵活与易用，并且能够和我们其他的优化算法相结合。
**Intel 套件：**
&ensp;&ensp;在使用 Intel 官方为我们提供的套件方面，我们主要用到了套件中的知识蒸馏，剪枝和量化。我们通过学习与运用这几个模型压缩方法的相关套件接口，实现了整个模型的精确度提升与检测时延的降低。
**识别优化：**
&ensp;&ensp;在思考如何进一步提高整个模型训练的速度与精度以达到自动驾驶所需的实时性和精确性时，我们考虑到了图片的高频分量这一因素。通过学习这一方面的内容，我们成功将其应用到我们的代码优化中，实现了目标图像边缘特性的检测提升，变相地提高目标检测的速度和精度。同时这启发了我们利用其他图像方面的知识进行模型优化的未来想法。
