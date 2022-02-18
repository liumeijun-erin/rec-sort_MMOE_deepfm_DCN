## P5 - 利用DCN实现电商场景下推荐

### 1.  DCN模型笔记

#### （1）input：连续型变量 + 离散型embedding

#### （2）resnet残差模块特征交叉：x‘ = x0xT w + b + x = f(x,w,b) + x

####   (3) 权重共享w 对X0XT进行降维 + 减少参数量

personal小结：

中间拟合残差，类似GDBT

增加cross，避免梯度反向传播时梯度消失，可以使深层模型更易训练

DeepCrosssing论文--提到归一化作用，在sample size变化时候表

X0与X隐形特征交叉（embedding交叉)，相对于FM来说提高交叉程度，相对于mlp来说减少参数

### 2. 实现笔记

Q1:cross结构实现 ：x0 传递；xi，xo迭代计算

Q2:如何实现层内w，b共享 -- 不能直接用dense + unit；需要tf计算公式实现

### 3. 在线部署与预估

* 方法1：C++ 读取pb+预估：C++ much faster

* 方法2：**docker部署 tf.serving**

  docker建立容器和本地saved_pd间映射；包装好数据进行请求

  