# Task1

第一版2024/10/17

## 思路

数据有13个维度，输出为1个维度

选用梯度下降算法，并选择小批量梯度下降

选择适合的cycle循环数避免过拟合情况等

选择合适的loss函数和优化器

## Prepare Dataset

数据为csv类型首先导入csv

由于数据处理较为麻烦，便直接设为一个class

#### 查看数据

```
data = pd.read_csv(file_path)

print(data.describe())
```

![img](https://raw.githubusercontent.com/Davereminisce/image/e3b0c4bde5f0df96f8e1c3ec8065e3e4d9849cf9/%7BABD20529-853C-4F24-8085-CAD5CE31B1DF%7D.png)

第一层为ID需要去掉（最终测试才发现的）

#### 异常处理

首先删除缺失值

使用替换异常值方法,使用四分位距法（IQR）来检测和替换异常值

#### 特征标准化和归一化

<img src="https://raw.githubusercontent.com/Davereminisce/image/f56966649cad556feaef67cc019724ccc670150e/%7BA12DED65-DE4C-460D-843A-ADDEC40208C5%7D.png" alt="img" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Davereminisce/image/339a4fdb541f6fe26bd0f29438335a36eaf399b3/%7B949E8322-3D1A-4D12-B8A2-97B3561FDA52%7D.png" alt="img" style="zoom:50%;" />

题目推荐对特征进行归一化处理。

但我认为在日常生活中的特征还是适用于正态分布并且选用的是线性回归所以选择**特征标准化**



#### 测试时发现问题

##### 类型错误

<img src="https://raw.githubusercontent.com/Davereminisce/image/de9998fadf658ac60d6df240944a8a654ed54e6b/%7B6441C07A-C913-4D30-8488-303CC60119AC%7D.png" alt="img" style="zoom:50%;" />

```
        self.x_data = data.drop(columns=['medv']).values    # 转换为 numpy 数组
        self.y_data = data['medv'].values                   # 转换为 numpy 数组

        #转化为torch tensors
        self.x_data = torch.tensor(self.x_data, dtype=torch.float32)
        self.y_data = torch.tensor(self.y_data, dtype=torch.float32)
```

这里data类型时pandas而后面处理的是numpy再转化torch tensors为所以需要转化数据类型

替换异常值方法需要numpy型

特征标准化也只能使用numpy型

再次运行测试任有问题

```
  File "D:\ML\Task1\Task1.py", line 26, in __init__
    for i in range(self.x_data.size(1)):
                   ^^^^^^^^^^^^^^^^^^^
TypeError: 'int' object is not callable
```

##### 调用错误导致命名冲突

因为代码中存在一个命名冲突

使用 `.shape` 而不是 `.size()`

这里改进后仍有问题

##### 多进程运行错误

发现，这里选择多进程执行进程为2，然而用的Windows系统，需要更改（查看机器学习笔记发现）

<img src="https://raw.githubusercontent.com/Davereminisce/image/1d91f7146aa87e637d514d774815231940e971a8/%7BA7EC25EF-7217-4AA1-9647-6D51D1306536%7D.png" alt="img" style="zoom:50%;" />

#### 测试代码

```
if  __name__ == '__main__':
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Data: {data}")
        print(f"Target: {target}")
        print("-" * 50)
```

结果

<img src="https://raw.githubusercontent.com/Davereminisce/image/5b0272899e1b644137df927dda63c0d195befb58/%7BE4D8D1B8-D973-47A5-AFEE-5CF83C047BA1%7D.png" alt="img" style="zoom:50%;" />

表示构建成功

## Define Model

这里选用线性模型torch.nn.Linear(,)

因为为多维特征，为了使其更有可变性，进行非线性变换设计加入激活函数

这里进行维度下降的设计为13-11-9-7-5-3-1

激活函数使用reLU

## Construct Loss and Optimizer

损失函数使用BCELoss二元交叉熵损失（错误！）

优化器使用SGD随机梯度下降

由最终测试发现上述损失函数不能在这使用（BCELoss用于二分类问题，要求模型的输出使用Sigmoid激活函数，将输出限制在0到1之间。），所以选择将损失函数改为MSE（均方误差）

## Training Cycle

之前数据处理时最后产生train_loader

需要读出x_data，y_data

```
for batch_idx, (x_data, y_data) in enumerate(train_loader):
```

```
#Training Cycle
if  __name__ == '__main__':
    for epoch in range(100):
        for batch_idx, (x_data, y_data) in enumerate(train_loader):
            pass

        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## GPU运算

这里linear过多，运算慢，于是设计为GPU运算

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
```

添加至model后

```python
inputs, target = inputs.to(device), target.to(device)
```

添加至需要传入GPU的数据后

## Train测试

### 测试错误

##### 测试出错：Model矩阵的形状不匹配

再次观察数据集发现有一行为ID数据处理没有去除所以放入x_data时需要同时除去medv和ID

```python
self.x_data = data.drop(columns=['medv']).values
#改为
self.x_data = data.drop(columns=['medv', 'ID']).values
```

##### 测试出错： y_pred和 y_data的维度不一致。

```python
# 训练循环中
y_pred = model(x_data)

# 添加以确保 y_data 维度与 y_pred 匹配
y_data = y_data.unsqueeze(1)
```

##### 测试出错：损失函数与优化器不匹配

因为使用了 BCELoss（二元交叉熵损失函数），但你的 y_data 中包含的值不在 [0, 1] 之间。

因为最终还是需要得出medv这个值不在[0,1]之间

所以该损失函数调整为MSE（均方误差）

### 运行

运行中途

<img src="https://raw.githubusercontent.com/Davereminisce/image/06bee3a196321642ed6c6ab2ea83b3a653dd06f4/%7BD7B205F6-6DE5-41E9-9048-2273F988BF5F%7D.png" alt="img" style="zoom:50%;" />

运行结果

<img src="https://raw.githubusercontent.com/Davereminisce/image/ab49ae80602ad90a6b85e84d009866e9cd7e56f5/%7BED173945-AA53-4763-9DA1-008FF92D445D%7D.png" alt="img" style="zoom:50%;" />

感觉误差还是很大

## 测试模块

Test函数

## 调整网络

#### 改动了epoch次数变为10，选择每次线性回归维度下降4（Model_4）

<img src="https://raw.githubusercontent.com/Davereminisce/image/f9f3627925464f3ce016fdc6f30a08ac9f7c648c/%7B49334F47-D280-4AB9-AA62-CBAC7EB79583%7D.png" alt="img" style="zoom:50%;" />

发现test有问题，由推测应该是Model出现问题

最后一层使用了 ReLU 激活函数，这可能会导致负值的输出被截断为 0。

删去最后一层 ReLU 激活函数

<img src="https://raw.githubusercontent.com/Davereminisce/image/e0561cab879369169377d964fd65e19a01fdc3df/%7B9A35777D-00FC-46E3-B3AB-F9C89F051991%7D.png" alt="img" style="zoom:50%;" />

此时输出值接进官网答案22.7687687687688

猜测是lr过大，改变lr优化输出界面

调试过程中test部分也出现问题

```
#Test                          
if  __name__ == '__main__':
    print('--------------Test--------------')
    with torch.no_grad():
        for idx, Test_data in enumerate(test_loader, 1):
            Test_data = data
            Test_data = Test_data.to(device)
```

这里那么 data 是一个列表，你需要将其转换为张量。

需要取出张量

```
    with torch.no_grad():
        for idx, (Test_data, _) in enumerate(test_loader, 1):
            Test_data = Test_data.to(device)

            output = model(Test_data)
            print(idx, output[0].item())
```

#### 降低lr->0.01

<img src="https://raw.githubusercontent.com/Davereminisce/image/d2635b0b9f11ba106884d260141832ad3886f9fe/%7B1988CB16-0C7D-40D7-99AD-A075CFC41C16%7D.png" alt="img" style="zoom:50%;" />

#### 改用LeakyReLU激活函数

torch.nn.LeakyReLU(negative_slope=0.01) 

```
6 416.41135025024414
7 615.0869255065918
8 407.00972175598145
9 418.62360668182373
10 732.2570743560791
--------------Test--------------
1 36.796592712402344
2 32.58344268798828
3 18.023902893066406
4 16.364429473876953
```

输出波动变大了

查阅资料有如下方式调整

调整学习率：尝试不同的学习率，观察对训练和测试结果的影响。

增加批量大小：如果当前批量大小较小，尝试增加到 32、64 或更高的值。

检查数据标准化：确保输入特征在相似的范围内。

监控训练过程：使用训练和验证集的损失值监控模型表现，及时调整。

#### 改变batch

由原来30-->64

```
6 778.7079925537109
7 540.7374839782715
8 401.7987766265869
9 406.0284423828125
10 452.58950424194336
--------------Test--------------
1 40.90989685058594
2 35.427940368652344
3 18.56787872314453
4 15.033670425415039
5 19.46322250366211
```

仍然不行

最终决定还是选用ReLU这样更接近同时降低每次线性维度下降数目使用Model_2

#### 用回ReLU降低每次维度下降数增加线性回归函数

```
7 494.6431007385254
8 517.5854072570801
9 538.2785415649414
10 490.7169418334961
--------------Test--------------
1 22.939167022705078
2 22.939167022705078
3 22.939167022705078
4 22.939167022705078
5 22.939167022705078
```

此时test非常接近了

## 重新训练同一模型测试结果差距较大现象

每次重新训练同一模型测试结果差距较大

第一次

<img src="https://raw.githubusercontent.com/Davereminisce/image/231e7537192439d00c7b0519abb8c0ab2db4ebdd/%7B9AE0506E-70EA-4E94-BF7C-9B8C0F860F41%7D.png" alt="img" style="zoom:50%;" />

第二次

<img src="https://raw.githubusercontent.com/Davereminisce/image/6335a60e20dacf25ffafa0ef15d8bb20ac7ddbd9/%7B391481E5-692C-4632-97F8-205D1586FACF%7D.png" alt="img" style="zoom:50%;" />

第三次

<img src="https://raw.githubusercontent.com/Davereminisce/image/d6baa0373ecd22fdc22fa85bf36e6bbd2600010f/%7B9A2882F9-75BF-4561-9C90-9398921DD27F%7D.png" alt="img" style="zoom:50%;" />

发现同一模型每次训练效果差距较大

查阅资料发现：

每次重新测试时结果差距较大是正常的，尤其在以下情况下：

1. **随机初始化**：每次训练模型时，权重的随机初始化可能会导致模型收敛到不同的局部最优解，尤其是在深度网络中。
2. **小批量随机性**：如果使用小批量梯度下降（mini-batch gradient descent），每个批次的样本随机性可能会导致训练过程中的波动。
3. **数据集的划分**：如果在不同的训练和测试集划分上测试，可能会导致结果的差异，尤其是当数据集较小或不均衡时。
4. **超参数的选择**：学习率、批量大小、正则化等超参数的选择会影响模型的训练过程，可能导致不同的收敛行为。
5. **噪声和过拟合**：数据中的噪声可能导致模型在不同训练迭代中的表现差异，特别是在过拟合的情况下。

应对方法

1. **多次实验**：多次运行实验并记录不同实验的结果，可以取平均值来评估模型的稳定性。
2. **交叉验证**：使用交叉验证方法来评估模型性能，这样可以减少因数据划分导致的结果波动。
3. **固定随机种子**：可以通过设置随机种子（如 `torch.manual_seed(seed)`）来确保每次训练的随机性相同，从而使结果更加一致。
4. **学习曲线**：绘制训练和验证的损失曲线，帮助观察模型的收敛趋势和稳定性。

###### 2024/10/17

## 优化模型（这是在Task1完成后对模型进行优化调整）

### Cycle结构阅读优化

建立train（），test（），最后统一调用

```python
if  __name__ == '__main__':
    train()
    test()
```

### model.train()&model.eval()

model.train()&model.eval()这两个是 PyTorch 中用于切换模型状态的方法。这两者的主要作用是在训练和评估（推理）之间切换模型的行为。

model.train()

将模型切换到训练模式。简单来讲就是说可以进行训练的运算。

用于训练阶段，启用特定行为以提高训练效果。

model.eval()

将模型切换到评估模式。简单来讲就是禁用梯度运算（相当于不用backward了）。

用于评估阶段，确保模型以正确的方式进行评估和推理。

### Cycle最佳循环数探究

测试Cycle_time

这里需要进行数据可视化

选用tensorboard（后面再深入）

2024/10/17

#### tensorboard引入

按照可视化.md加入tensorboard发现成功可视化，但是会产生多个文件夹产生多组日志文件，不断测试（不断调整都快红温了......）

最后发现是由于多线程问题，将多线程删除后便只产生一个日志文件了。

##### 多线程或多进程下tensorboard引发的问题

在多线程或多进程下，在每个线程、进程都独立执行并且都有创建SummaryWriter实例的逻辑下，会导致每个线程、进程都创建自己的日志目录。这是因为每个实例在初始化时都会根据提供的路径生成一个新的目录。

**线程/进程独立性**：每个线程或进程都有自己的执行上下文，它们不会共享内存或变量。如果在每个线程/进程中都调用了`SummaryWriter`，每个调用都会根据`log_dir`创建一个新的日志目录。

**初始化逻辑**：如果`SummaryWriter`的创建放在了多线程或多进程的代码块中，那么每个执行这个代码块的线程/进程都会触发目录创建。

**示例**：比如在训练中使用`DataLoader`并且设置`num_workers`参数时，PyTorch会使用多进程来加载数据。如果在数据加载过程中初始化了`SummaryWriter`，每个工作进程都会创建一个新的日志目录。

###### 解决（解决多进程引起的日志问题）

为避免这种情况，确保`SummaryWriter`只在主进程中创建，并且传递给需要记录日志的函数，而不是在多线程或多进程中重复创建。

只有在DataLoader的num_workers参数大于0时，才会引入多进程。

仅需简单操作即可（红温了）

第一种：删除多线程，最后发现仍然有，因为仍然存在多进程但是少了许多日志文件。

第二种：将writer = SummaryWriter(log_dir)放到主进程

```python
if  __name__ == '__main__': 
    writer = SummaryWriter(log_dir)	#放这里来
    print(f'Logging to: {log_dir}')
    train()
    test()
    writer.close()
```

##### 可视化展示

loss

<img src="https://raw.githubusercontent.com/Davereminisce/image/267a9eca910c28edc47c26aa79f8df804a31538c/%7BAB79EA23-D7A5-471C-9F11-32057C170056%7D.png" alt="img" style="zoom: 33%;" />

test predicted price

<img src="https://raw.githubusercontent.com/Davereminisce/image/99357a3554342722f949e67de968e804481b4bd1/%7BC3051DEB-B890-4653-BB15-037097E6240B%7D.png" alt="img" style="zoom:33%;" />

### nan问题

这里如果训练时出现nan是由于训练过程中数值不稳定引起的，可以进行低学习率操作
