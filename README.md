# CIFAR
## 项目简介
CIFAR是由Alex Krizhevsky、Vinod Nair和Geoffrey Hinton收集而来，起初的数据集共分10类，分别为飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车，所以CIFAR数据集常以CIFAR-10命名。CIFAR共包含60000张32*32的彩色图像（包含50000张训练图片，10000张测试图片），其中没有任何类型重叠的情况。因为是彩色图像，所以这个数据集是三通道的，分别是R，G，B3 个通道。后来CIFAR又出了一个分类更多的版本叫CIFAR-100，共有100类，将图片分得更细，当然对神经网络图像识别是更大的挑战了。有了这些数据，我们可以把精力全部投入在网络优化上。

[CIFAR地址](http://www.cs.toronto.edu/~kriz/cifar.html)

采用CNN网络完成训练，来探讨网络框架的性能。
## 文件结构
```
    .
    |—— data #CIFAR数据集
    |—— train #保存的模型
    |—— log #tensorboard保存的文件地址
    *.py #代码文件
    Makefile
    README.md
    environment.yaml
    .gitignore
```

## 环境配置
```
conda env create -f environment.yaml
```

## 数据集获取
```
cd .
make data
```

## 运行
```
cd .
python main.py --mode cifar10/cifar100
```
