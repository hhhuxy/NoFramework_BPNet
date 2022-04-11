## 运行环境

* 使用anaconda进行环境配置，在conda命令行中输入以下代码以安装所需环境

  ````
  conda env create -f env.yml
  ````

  激活环境：

  ````
  conda activate digit_recognition
  ````

  运行程序：

  `python main.py ` （该命令以默认参数运行程序）若需修改参数，可参照以下命令进行修改：
  
  ````bash
  python main.py --quiet False --epoch 20 --init_method uniform --init_bound -0.5 0.5 --optimizer minibatch --layer_sizes 784 100 10 --print_score True --activation_func sigmoid
  ````
  
  选项具体细节请参考`文档.pdf`1.3节
  
  退出环境：
  
  ````
  conda deactivate
  ````

## 文件说明

`bp_model`: BP神经网络模型无框架python实现

`main.py`：运行手写数字识别main文件

`mnist.pkl.gz`：MINIST数据集

`env.yml`：环境配置文件

`文档.pdf`：项目文档

`运行截图.png`，`运行截图2.png`：程序运行截图


本项目的识别精度f1分值可达约97，随参数规模增加可获得进一步提升，但会导致训练速度较慢。


Author: Xinyun Hu, Beijing University of Posts and Telecommunications

2022.4
