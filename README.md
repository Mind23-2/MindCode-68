# 目录

<!-- TOC -->

- [ResNet3D描述](#ResNet3D描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# glore_res描述

## 概述

Video Classification：
在大型数据集Kinetics-400, Kinetics-700以及合成的数据集Kinetics-700 + MiT上预训练模型，之后在小型数据集UCF-101，HMDB-51, ActivityNet上进行微调，相比直接在上述几个小型数据集上训练，得到的分类精度得到了提升。


如下为MindSpore使用UCF101,HMDB51数据集对ResNet3D进行fine-tune训练的示例。ResNet3D可参考[论文1](https://arxiv.org/pdf/2004.04968v1.pdf)

## 论文

1. [论文](https://arxiv.org/pdf/2004.04968v1.pdf)：Hirokatsu Kataoka, Tenga Wakamiya, Kensho Hara, Yutaka Satoh "Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs?"

# 模型架构

ResNet3D的总体网络架构如下：
[链接](https://arxiv.org/pdf/2004.04968v1.pdf)

# 数据集

使用的数据集：
[UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/)

- 数据集大小：共101个类、共13,320个320*240视频文件
- 数据格式：AVI
- 下载数据集，目录结构如下：

 ```text
└─UCF101
    ├─video                # 数据集
    └─split                # 标注文件.txt
```
[HMDB51](https://www.crcv.ucf.edu/research/data-sets/ucf101/)
- 数据集大小：共51个类、共6849个320*240视频文件
- 数据格式：AVI
- 下载数据集，目录结构如下：

 ```text
└─HMDB51
    ├─video                # 数据集
    └─split                # 标注文件.txt
```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件(Ascend)
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```text
# 分布式训练
用法：sh run_distribute_train.sh [DATA_PATH] [DEVICE_NUM]

# 单机训练
用法：sh run_standalone_train.sh [DATA_PATH] [DEVICE_ID]

# 运行评估示例
用法：sh run_eval.sh [DATA_PATH] [DEVICE_ID] [CKPT_PATH]
```

# 脚本说明

## 脚本及样例代码

```shell
.
└──glore_res50
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_eval.sh                        # 启动Ascend评估
    ├── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
  ├── src
    ├── __init__.py
    ├── autoaugment.py                     # AutoAugment组件与类
    ├── config.py                          # 参数配置
    ├── dataset.py                         # 数据预处理
    ├── glore_res50.py                     # glore_res50网络定义
    ├── loss.py                            # ImageNet2012数据集的损失定义
    ├── save_callback.py                   # 训练时推理并保存最优精度下的参数
    └── lr_generator.py                    # 生成每个步骤的学习率
  ├── eval.py                              # 评估网络
  └── train.py                             # 训练网络
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置glore_res50和ImageNet2012数据集。

```text
"class_num":1000,                # 数据集类数
"batch_size":128,                # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":120,                # 此值仅适用于训练；应用于推理时固定为1
"pretrained": False,             # 加载预训练权重
"pretrain_epoch_size": 0,        # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点相对于执行路径的保存路径
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"Linear",        # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.05,      # 标签平滑因子
"weight_init": "xavier_uniform",      # 权重初始化方式,可选"he_normal", "he_uniform", "xavier_uniform"
"use_autoaugment": True,         # 是否应用AutoAugment方法
"lr_init":0,                     # 初始学习率
"lr_max":0.8,                    # 最大学习率
"lr_end":0.0,                    # 最小学习率
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```text
# 分布式训练
用法：sh run_distribute_train.sh [DATA_PATH] [DEVICE_NUM]

# 单机训练
用法：sh run_standalone_train.sh [DATA_PATH] [DEVICE_ID]

```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)中的说明。

### 结果

- 使用ImageNet2012数据集训练glore_res50

```text
# 分布式训练结果（8P）
epoch:1 step:1251, loss is 5.721338
epoch:2 step:1251, loss is 4.8941164
epoch:3 step:1251, loss is 4.3002024
epoch:4 step:1251, loss is 3.862403
epoch:5 step:1251, loss is 3.5204496
...
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 评估
用法：sh run_eval.sh [DATA_PATH] [DEVICE_ID] [CKPT_PATH]
```

```bash
# 评估示例
sh run_eval.sh ~/dataset/imagenet 0 ~/ckpt/glore_res50_120-1251.ckpt
```

### 结果

评估结果保存在示例路径中，文件夹名为“eval”。您可在此路径下的日志找到如下结果：

- 使用ImageNet2012数据集评估glore_res50

```text
{'Accuracy': 0.7844638020833334}
```

# 模型描述

## 性能

### 评估性能

#### ImageNet2012上的glore_res50

| 参数                 | Ascend 910
| -------------------------- | -------------------------------------- |
| 模型版本              | glore_res50
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期              | 2021-03-21 |
| MindSpore版本          | r1.1 |
| 数据集                    | ImageNet2012 |
| 训练参数        | epoch=120, steps per epoch=1251, batch_size = 128 |
| 优化器                  | Momentum |
| 损失函数              | Softmax交叉熵 |
| 输出                    | 概率 |
| 损失                       | 1.8464266 |
| 速度                      | 263.483毫秒/步（8卡）|
| 总时长                 | 10.98小时 |
| 参数(M)             | 30.5 |
| 微调检查点| 233.46M（.ckpt文件）|
| 脚本                    | [链接](https://gitee.com/mindspore/mindspore/tree/glore_res50_r1.1/model_zoo/research/cv/glore_res50) |

# 随机情况说明

使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
