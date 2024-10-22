# Task2（图像分类）

这个可以参照机器学习基础中的卷积神经网络CNN

## 思路

将图像进行数据处理导入

图像尺寸为32x32，分为50,000张训练图像和10,000张测试图像。

(这里还要注意图像的通道数)

还是使用mini batch小批量运算

Model使用卷积神经网络使用多通道卷积构建多个卷积核

卷积构建torch.nn.Conv2d()

池化torch.nn.MaxPool2d()

#### 图像通道数计算&分类的情况

<img src="https://raw.githubusercontent.com/Davereminisce/image/0a67b91c64b0b3fbe622813c5d929e9a8e66871b/%7BECAAEA32-AC16-499F-A9E1-214C256DD663%7D.png" alt="img" style="zoom:50%;" />

channel = 3

num_classes = 10

分类名称： ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

## Model（Network）

整体思路：初始数据集->卷积->激活->卷积->激活->卷积->池化->激活......

先 （卷积->激活->卷积->激活->）提取捕捉更复杂的特征

再 （卷积->池化->激活->）保留主要特征，同时去除一些不必要的细节，有助于提高模型的泛化能力。

### 四种函数

self.conv1 = torch.nn.Conv2d(in，out，kernel_size)

self.pooling = torch.nn.MaxPool2d(窗口size)

x.view（batch_size，-1）

self.fc = torch.nn.Linear（卷积后数据维度，目标输出维度）

### 流程

（batch，3，32，32）

卷积层1	|---->Conv2d(3，10，kernel_size = 3)

激活层	  |---->rulu

（batch，10，30，30）

卷积层2	|----> Conv2d(10，15，kernel_size = 3)

激活层	  |---->rulu

（batch，15，28，28）

卷积层3	|---->Conv2d(15，20，kernel_size = 5)

激活层	  |---->rulu

（batch，20，24，24）

池化层	  |----> MaxPool2d(2)

（batch，20，12，12）

卷积层4	|---->Conv2d(20，25，kernel_size = 5)

激活层	  |---->rulu

（batch，25，8，8）

池化层	  |----> MaxPool2d(2)

（batch，25，4，4）

变为线性      |---->view（batch_size，-1）

（batch，400）

线性回归      |---->Linear（400，10）

（batch，10）

<img src="https://raw.githubusercontent.com/Davereminisce/image/51f001ebb1804a56b138316624e248a6e56d9ad6/%7BA956A8F1-8600-4ACE-8B27-08AE8DDCB4FE%7D.png" alt="img" style="zoom:50%;" />

## Loss and Optimizer

#### Loss选用

因为这个模型是将图像进行分类从而达到识别的目的，所以选用交叉熵损失（Cross Entropy Loss）

#### Optimizer选用

选用对于图像分类常用的Adam

## 阅读框架发现

### model.eval()

```python
def test():
    model.eval()        
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
       		pass
```

#### 解释

在进行模型评估、验证或推理时，调用 model.eval()可以确保模型在评估时使用正确的行为。

#### 注意

1.在调用model.eval()后，通常会用 with torch.no_grad(): 包裹评估代码，以禁用梯度计算，从而减少内存使用和提高计算效率。

2.在完成评估后，如果要继续训练模型，要调用 model.train()将模型切换回训练模式。

###  with torch.no_grad():

#### 解释

是 PyTorch 中用于禁用梯度计算的上下文管理器。它的主要作用是在不需要计算梯度的情况下进行前向传播，从而节省内存和提高计算效率。

#### 注意

1.这是一个块

2.仅在块内所有的张量操作都不会计算梯度。该块结束后，梯度计算会恢复正常。

## 运行结果

<img src="https://raw.githubusercontent.com/Davereminisce/image/975f3afaf767375d3e9522a61ed8785555571947/%7B2F1EC7E5-1651-47CC-8335-3E71F834C1CB%7D.png" alt="img" style="zoom:50%;" />

# Task2TB（可视化）

参考训练过程可视化工具.md文档，

#### 导入tensorboard

```python
from torch.utils.tensorboard import SummaryWriter
```

#### 设置日志路径

```python
log_dir = r'Task2\LogExtra1' 
writer = SummaryWriter(log_dir)
```

#### 录入数据到日志

```python
writer.add_scalar('loss/epoch', running_loss, epoch)
writer.add_scalar('accuracy/epoch', accuracy, epoch)       
```

#### 训练结束关闭SummaryWriter

```python
writer.close()
```

#### 终端启动TensorBoard

```bash
tensorboard --logdir=D:\ML\Task2\LogExtra1 --port=6001
```

### 运行结果

#### 训练途中

<img src="https://raw.githubusercontent.com/Davereminisce/image/9b22e887724c31a4007919122c222f8a98f92542/%7B55400FB8-26EF-43C5-8A38-68DB06A979A1%7D.png" alt="img" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/Davereminisce/image/3f010791ad3065ba86fe27fb5b7cccf56f1ec662/%7B6900CE19-82C3-4946-A83D-567711265C08%7D.png" alt="img" style="zoom:80%;" />

#### 训练结束

<img src="https://raw.githubusercontent.com/Davereminisce/image/f7c69158f19c0a1d6db65f9e7bf48af56cbe4ac5/%7BAC08A415-BACE-4B7B-A051-A7CAB75BF886%7D.png" alt="img" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/Davereminisce/image/1162076bf16a2bdd7484c0093d13cc14ddeed866/%7BD9A9B763-8632-45B3-8017-4027EAA0BDB2%7D.png" alt="img" style="zoom:50%;" />

### 模型代码优化

#### 调整test和train，使每次epoch都进行test并计算出正确率

<img src="https://raw.githubusercontent.com/Davereminisce/image/7fcf4915ba86bd620af454ffd9db40afc57936e7/%7B4C56567A-0CE2-4D8D-829F-9570E68F770B%7D.png" alt="img" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Davereminisce/image/e51fdb121e7cc05c48599b955eb7a5d3c9a1f09c/%7B3C02FAA5-842B-4CFB-BD52-4DEC1C7C5343%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/71112a2260f11ae56f5b29cb7ddd3dda4cb3500f/%7BFF710399-6661-4729-8D93-70882574D368%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/9914f7eaf883be32506485faf63bc548d75b43db/%7B8F190560-9D3E-46BC-A944-EFA8E6D41AA2%7D.png" alt="img" style="zoom:33%;" />



# Task2GNbasic（GoogleNet）

这里只是简单的运用其中的Inception块，来自己构建GoogleNet（虽然效果有点不理想）但是比普通卷积好点

#### Inception块

<img src="https://raw.githubusercontent.com/Davereminisce/image/f7a4c5c16020716397719adf17ec792a1375b91d/%7B12B2783D-AB50-4FB3-A4D8-0184BF2D99E5%7D.png" alt="img" style="zoom:50%;" />

输入（批大小b，通道数c，w，h）

##### branch1(branch_pool)

(b,c,w,h)

-->avg_pool2d(x,kernel_size=3,stride=1,padding=1)-->(b,c,w,h)

-->Conv2d(c,24,kernel_size=1)-->(b,24,w,h)

##### branch2(branch_1x1)

(b,c,w,h)

-->Conv2d(c,16,kernel_size=1)-->(b,16,w,h)

##### branch3(branch_5x5)

(b,c,w,h)

-->Conv2d(c,16,kernel_size=1)-->(b,16,w,h)

-->Conv2d(16,24,kernel_size=5,padding=2)-->(b,24,w,h)

##### branch4(branch_3x3)

(b,c,w,h)

-->Conv2d(c,16,kernel_size=1)-->(b,16,w,h)

-->Conv2d(16,24,kernel_size=3,padding=1)-->(b,24,w,h)

-->Conv2d(24,24,kernel_size=3,padding=1)-->(b,24,w,h)

##### 合并branch

outputs = [branch1,branch2,branch3,branch4]

x = torch.cat(outputs,dim=1)

(b,88,w,h)

#### model块

改进后

（b,3,32,32）

-->Conv2d1(3,16,kenel_size=3)-->ReLU-->(b,16,30,30)

-->Conv2d2(16,64,kenel_size=3)-->MaxPool(2)-->ReLU-->(b,64,14,14)

-->Inception1(in_channels=64)-->(b,88,14,14)

-->Conv2d3(88,64,kenel_size=3)-->MaxPool(2)-->ReLU-->(b,64,6,6)

-->Inception2(ini_channels=64)-->(b,88,6,6)

-->ReLU-->(b,88,6,6)

-->x.view(x.size(0),-1)-->(b,3168)

-->Linear(4608,10)-->(3168,10)

### 日志目录优化

```
import time
log_dir = f'D:/ML/Task2/logs/{int(time.time())}'
```

```
print(f'Logging to: {log_dir}')
```

## Task2GNbasic运行结果

batch_size = 16

learning_rate = 0.001

num_epochs = 20

<img src="https://raw.githubusercontent.com/Davereminisce/image/8ef3052825742fea48b82434d95427b1bff791a9/%7BD90B3CFB-95DC-4C1F-93EE-9C85C7F7D3C2%7D.png" alt="img" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Davereminisce/image/48bf6d44551c10573bb4694a25ce88f17d8d46ee/%7BA46DF9DB-2368-4D83-8A9A-6AF85AF32A51%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/53e9586f104b7164f182057fd0f3f422a1cfb675/%7B0232BB95-9E97-4B85-9290-AE7A47E3692B%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/b2c5bdaaef0112b1477d3d65d9a6290a536fa26d/%7BCC013DB2-D651-44C9-9E14-42871B96403B%7D.png" alt="img" style="zoom:33%;" />

# Task2RNbasic（ResNet）

这里学习残差块构建并自己设计resnet，效果与GNbasic差不多

#### 残差块(ResidualBlock)

一个典型的残差块包含以下部分：

两个卷积层，每个卷积层后面跟着批量归一化（Batch Normalization）和 ReLU 激活函数。

输入x直接通过跳跃连接与残差函数 F(x) 的输出相加。

输入（批大小b，通道数c，w，h）

（b, c, w, h）0

->conv(c, c, kernel_size=3, padding=1)->relu->(b, c, w, h)1

->conv(c, c, kernel_size=3, padding=1)->(b, c, w, h)2

(b, c, w, h)0+(b, c, w, h)2->(b, c, w, h)->relu

->(b, c, w, h)

#### Net构建

（b, 3, 32, 32）

->conv1(3, 8, kernel_size=3)->relu->(b, 8, 30, 30)

->conv2(8, 16, kernel_size=3)->MaxPool2d(2)->relu->(b, 16, 14,14)

->ResidualBlock1(16)->(b, 16, 14, 14)

->conv3(16, 20, kernel_size=3)->MaxPool2d(2)->relu->(b, 20, 6, 6)

->ResidualBlock2(20)->(b, 20, 6, 6)

-->x.view(x.size(0),-1)-->(b,720)

-->Linear(720,10)-->(b,10)

## Task2RNbasic运行结果

batch_size = 32

learning_rate = 0.001

num_epochs = 40

<img src="https://raw.githubusercontent.com/Davereminisce/image/4e3f5b197e09b4ae84ad50c3de19710b85dd97b3/%7B13C7E3D8-EF22-4A8D-997F-F5EC0295A4D2%7D.png" alt="img" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Davereminisce/image/57e6bf770f14f3e36d7a671f5aa0ea4f17faba08/%7BE398B41F-B455-4280-AEDA-9A3072C0A0E1%7D.png" alt="img" style="zoom: 33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/c323ca07b405a177c3624f52e7cee5dd8c190a64/%7BB62B62F6-095E-47E9-B9AB-F459E3270BE3%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/73b24410bcbafdaaf5f1d274bcbed2783955ae78/%7B796CB07A-9FE9-4770-80C1-5A3DDEEEF5B5%7D.png" alt="img" style="zoom:33%;" />

# Task2GoogleNet.py

这里发现上述准确率还是很低，于是去学习了更加完整的GoogleNet

对于原Task2的更改已经在Task2GN.py中标志出了

首先用cv2.resize增大了原数据的像素数

增大了inception块中的参数

在Net构建中增大了inception块的堆叠数量

这里参数设置以及代码编写借用了csdn上的一篇[相关文章]https://blog.csdn.net/qq_51957239/article/details/129380171)

然后在Task2的基础上进行复现

### GoogleNet学习：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5ba1b1224d9066d621ef6d57091f0e77.png)

结构
卷积层，输入为224 × 224的彩色图像，使用7 × 7的卷积核，步幅为2，输出通道数为64，偏置和ReLU激活函数。
最大池化层，使用3 × 3的卷积核，步幅为2，输出尺寸为112 × 112。
卷积层，输入通道数为64，使用1 × 1的卷积核，输出通道数为64，偏置和ReLU激活函数。
卷积层，输入通道数为64，使用3 × 3的卷积核，输出通道数为192，步幅为1，偏置和ReLU激活函数。
最大池化层，使用3 × 3的卷积核，步幅为2，输出尺寸为56 × 56。
使用两个Inception模块，输出通道数为256和480，分别将它们堆叠在一起。
最大池化层，使用3 × 3的卷积核，步幅为2，输出尺寸为28 × 28。
使用五个Inception模块，输出通道数分别为512 , 512 , 512 , 528和832，分别将它们堆叠在一起。
最大池化层，使用3 × 3的卷积核，步幅为2，输出尺寸为14 × 14。
使用两个Inception模块，输出通道数分别为832和1024，分别将它们堆叠在一起。
全局平均池化层，使用一个7 × 7的池化窗口，输出大小为1 × 1。
Dropout层，随机将一定比例的元素归零，防止过拟合。
全连接层，输出为10，对应10个类别。

### 实验记录

batch_size = 32

learning_rate = 0.0001 

num_epochs = 20  

<img src="https://raw.githubusercontent.com/Davereminisce/image/4ec6263466249068ae63564507fa13d3b1e46dc0/%7B6E6A69AE-FD4B-40A5-82F9-5DBC9319D0E2%7D.png" alt="img" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Davereminisce/image/04d2c0d33c062b9873d2fdd25bc5923451c97951/%7B0DDE41B1-7133-4ED6-AE03-C1CA812C80A2%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/bfd68cd414d720202d2b0f626a273d8fa0899f23/%7B37E92841-1E75-4760-A6BF-C057D1FF4256%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/ff15795b633549c8f4e27c272ea950cc7af59050/%7B90D4ECC4-6D98-4227-AC6B-10AE5D9FB5CB%7D.png" alt="img" style="zoom:33%;" />

预计调整lr和增加epoch还会增加

# Task2ResNet18.py

这里直接用了预处理的ResNet18模型在原本的Task2.py基础上进行复现

### 实验记录

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

batch_size = 32

learning_rate = 0.001

num_epochs = 10

<img src="https://raw.githubusercontent.com/Davereminisce/image/6605454aa789310f048dd29ff279438c8bbed7ab/%7B03AC9761-9CD8-4557-84CE-33160B2FC759%7D.png" alt="img" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Davereminisce/image/c7391bbf8b8c8115ae714c35feedd8c63845adff/%7B12F72EC0-7DDC-473B-B39F-E1D1FEF5A99E%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/29c26d6b3211fce96e6b5cfd722496915d43dca8/%7B1C18B272-82DC-475B-8220-469F2885CAB5%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/433e252aa9b3c786dd3322d73fa7ed8f446d2942/%7B39BF868A-2B93-4132-9D41-93759FF39EB5%7D.png" alt="img" style="zoom:33%;" />

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

batch_size = 32

learning_rate = 0.0001

num_epochs = 20

<img src="https://raw.githubusercontent.com/Davereminisce/image/60e889c627c04191811d6186ad42747cec006757/%7BA4DD321D-AF4F-42B5-8372-9313CA48B979%7D.png" alt="img" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Davereminisce/image/8f1ca2114759ff68195f66c4482f492273057519/%7BECD4DCF7-BEDF-4F74-B831-7A936AB0F00C%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/bf8b8296ea048e105b5b0537739277ff63ea3490/%7BD2B3134B-DED2-4B63-B030-8F1D09005D69%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/0224c8ec78ce6d693c6ec67bddfb3536aaf86651/%7B0343B713-7DEB-4964-87EE-C4F5DC13E2C2%7D.png" alt="img" style="zoom:33%;" />
