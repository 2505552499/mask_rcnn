# Mask R-CNN

## 该项目参考自pytorch官方torchvision模块中的源码(使用pycocotools处略有不同)
* https://github.com/pytorch/vision/tree/master/references/detection

## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.10或以上
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`(不需要额外安装vs))
* Ubuntu或Centos(不建议Windows)
* 最好使用GPU训练
* 详细环境配置见`requirements.txt`

## 文件结构：
```
  ├── backbone: 特征提取网络
  ├── network_files: Mask R-CNN网络
  ├── train_utils: 训练验证相关模块（包括coco验证相关）
  ├── my_dataset_coco.py: 自定义dataset用于读取COCO2017数据集
  ├── my_dataset_voc.py: 自定义dataset用于读取Pascal VOC数据集
  ├── train.py: 单GPU/CPU训练脚本
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── transforms.py: 数据预处理（随机水平翻转图像以及bboxes、将PIL图像转为Tensor）
```

## 预训练权重下载地址（下载后放入当前文件夹中）：
* Resnet50预训练权重 https://download.pytorch.org/models/resnet50-0676ba61.pth (注意，下载预训练权重后要重命名，
比如在train.py中读取的是`resnet50.pth`文件，不是`resnet50-0676ba61.pth`)
* Mask R-CNN(Resnet50+FPN)预训练权重 https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth (注意，
载预训练权重后要重命名，比如在train.py中读取的是`maskrcnn_resnet50_fpn_coco.pth`文件，不是`maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth`)
 
 

## 数据集准备
```
├─datasets: 数据集根目录
     └── images: 
         |── annotaions: 存放标注文件
              ├── instances_train.json: 对应目标检测、分割任务的训练集标注文件
              ├── instances_val.json: 对应目标检测、分割任务的验证集标注文件
         |── train: 训练集图片
         |── val: 验证集图片
         |── test: 测试集图片
         |── result:预测的结果图片
             
```



## 训练方法
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 确保设置好`--num-classes`和`--data-path`
* 若要使用单GPU训练直接使用train.py训练脚本
* 若要使用多GPU训练，使用`torchrun --nproc_per_node=8 train_multi_GPU.py`指令,`nproc_per_node`参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上`CUDA_VISIBLE_DEVICES=0,3`(例如我只要使用设备中的第1块和第4块GPU设备)
* `CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py`

## 注意事项
1. 在使用训练脚本时，注意要将`--data-path`设置为自己存放数据集的**根目录**：
```
# 假设要使用COCO数据集，启用自定义数据集读取CocoDetection并将数据集解压到成/data/coco2017目录下
python train.py --data-path /data/coco2017

# 假设要使用Pascal VOC数据集，启用自定义数据集读取VOCInstances并数据集解压到成/data/VOCdevkit目录下
python train.py --data-path /data/VOCdevkit
```

2. 如果倍增`batch_size`，建议学习率也跟着倍增。假设将`batch_size`从4设置成8，那么学习率`lr`从0.004设置成0.008
3. 如果使用Batch Normalization模块时，`batch_size`不能小于4，否则效果会变差。**如果显存不够，batch_size必须小于4时**，建议在创建`resnet50_fpn_backbone`时，
将`norm_layer`设置成`FrozenBatchNorm2d`或将`trainable_layers`设置成0(即冻结整个`backbone`)
4. 训练过程中保存的`det_results.txt`(目标检测任务)以及`seg_results.txt`(实例分割任务)是每个epoch在验证集上的COCO指标，前12个值是COCO指标，后面两个值是训练平均损失以及学习率
5. 在使用预测脚本时，要将`weights_path`设置为你自己生成的权重路径。
6. 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时需要修改`--num-classes`、`--data-path`、`--weights-path`以及
`--label-json-path`（该参数是根据训练的数据集设置的）。其他代码尽量不要改动


## 复现结果
在COCO2017数据集上进行复现，训练过程中仅载入Resnet50的预训练权重，训练26个epochs。训练采用指令如下：
```
torchrun --nproc_per_node=8 train_multi_GPU.py --batch-size 8 --lr 0.08 --pretrain False --amp True
```

训练得到权重下载地址： https://pan.baidu.com/s/1qpXUIsvnj8RHY-V05J-mnA  密码: 63d5

在COCO2017验证集上的mAP(目标检测任务)：
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.381
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.588
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.411
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.215
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.315
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.666
```

在COCO2017验证集上的mAP(实例分割任务)：
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.340
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.552
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.361
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.151
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.290
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.468
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.266
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.509
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.619
```


