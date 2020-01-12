# ssd_pytorch  
  
首先发下代码的参考连接 [ssd-pytorch](https://github.com/amdegroot/ssd.pytorch)    
  
运行环境：  
>1.python 3.7.4  
>2.pytorch 1.3.1  
>3.python-opencv  
  
## 说明 
本模型是利用ssd卷积神经网络，训练了一个对安检照片中危险品（充电宝）的目标检测的模型。  
具体的配置文件请看Config.py文件  
训练运行Train.py   
测试运行Test.py 
计算map运行calculate_map_test.py 
  
  
## 注：  
>1.目前的版本是作者学习ssd模型并参考他人代码尝试复现的结果，所以代码鲁棒性较低，望见谅  
>2.目前版本只兼容voc的数据集  


