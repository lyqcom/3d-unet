# CadiacSeg3D.MindSpore
# Content 

- [3DUNet Descriptions](#3dunet-descriptions)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
- [Script Parameters](#script-parameters)
- [Training Process](#training-process)
- [Evaluation Process](#evaluation-process)
  - [Evaluation](#evaluation)
- [Model Description](#model-description)
  - [Performance](#performance)
    - [Training Performance](#training-performance)
    - [Inference Performance](#inference-performance)



## 3DUNet Descriptions

3DUNet was proposed in 2016, it is a type of neural network that directly consumes volumetric images. The 3DUNet extends the previous u-net architecture by replacing all 2D operations with their 3D counterparts. The implementation performs on-the-fly elastic deformations for efficient data augmentation during training. It is trained end-to-end from scratch, i.e., no pre-trained network is required.


[Paper](https://arxiv.org/pdf/1606.06650.pdf): Çiçek, Özgün, et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation." *International conference on medical image computing and computer-assisted intervention*. Springer, Cham, 2016.


## Model Architecture

The 3DUNet segementation network takes n 3D volumetric images as input, applies input and feature transformations. BN is introdued before each ReLU. 


## Dataset

Dataset used: [MM-WHS](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/), 3d-Covid-segmentation.

MM-WHS 

* Dataset size: 1.2G
  * Train: 428M, 20 images and corresponding labels.
  * Test: 761M
* Data format: NifTI files
  * Notes: Data will be processed in src/dataset.py


Covid-19 CT image dataset （3D）

Dataset size: 3.5G

* Train: 2.8G, 160 images and corresponding labels.
* Test: 724M,  49 images and corresponding labels.

## Environment Requirements

- Hardware（Ascend）
  - Prepare hardware environment with Ascend processor.
- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

* suwen package

  ```bash
  pip install -r requirements.txt
  pip install ./suwen-1.0.1-py3-none-any.whl
  ```

  

## Quick Start

After installing MindSpore via the official website, you can start training and evaluation as follows:

```python
# enter script dir, train PointNet
sh run_train_ascend.sh

# enter script dir, evaluate PointNet
sh run_eval.sh
```



## Script Description

```
.
└── 3dUNet
    ├── README.md
    ├── UNet3d.ckpt
    ├── eval.py
    ├── eval_log.txt
    ├── requirements.txt
    ├── scripts
    │   ├── run_eval.sh
    │   └── run_train_ascend.sh
    ├── src
    │   ├── __init__.py
    │   ├── config.py
    │   ├── convert_nifti.py
    │   ├── dataset.py
    │   ├── dice_metric.py
    │   ├── loss.py
    │   ├── lr_schedule.py
    │   ├── transform.py
    │   └── utils.py
    ├── suwen-1.0.1-py3-none-any.whl
    ├── train.py
    └── train_log.txt
```



## Script Parameters

```
Major parameters in train.py are as follows:

--data_path: The absolute full path to the train and evaluation datasets.
--seg_path : The absolute full path to the train and evaluation segmentation labels.
--ckpt_path: The absolute full path to the checkpoint file saved after training.
```

More hyperparamteters can be modified in src/config.py.



## Training Process

* running on Ascend

  ```
  sh run_train_ascend.sh
  ```

  After training, the loss value will be achieved as what in train_log.txt

  The model checkpoint will be saved in the current ckpt directory.
  
  
## Evaluation Process

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

  ```
  sh scripts/run_eval.sh
  ```
  
  You can view the results through the file "eval_log". The accuracy of the test dataset will be as what in eval_log.txt.
  
  

## Model Description

### Performance

#### Training Performance

| Parameters                 |                                               |
| -------------------------- | --------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 24cores; Memory, 96G |
| uploaded Date              | 11/15/2021 (month/day/year)                   |
| MindSpore Version          | 1.3.0                                         |
| Dataset                    | MM-WHS                                        |
| Training Parameters        | epoch=600, steps=, batch_size = , lr=         |
| Optimizer                  | Adam                                          |
| Loss Function              | Softmax Cross Entropy                         |
| outputs                    | probability                                   |
| Loss                       | SoftmaxCrossEntropyWithLogits                 |
| Speed                      | 212.469 ms/step-                              |
| Total time                 | 3399.497 ms                                   |
| Checkpoint for Fine tuning | 56M (.ckpt file)                              |

#### Inference Performance

| Parameters        |                                               |
| ----------------- | --------------------------------------------- |
| Resource          | Ascend 910; CPU 2.60GHz, 24cores; Memory, 96G |
| uploaded Date     | 05/29/2021 (month/day/year)                   |
| MindSpore Version | 1.3.0                                         |
| Dataset           | MM-WHS                                        |
| batch_size        | 1                                             |
| outputs           | probability                                   |
| Dice              | 85.08%                                        |
