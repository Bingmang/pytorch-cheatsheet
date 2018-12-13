# pytorch-cheatsheet

## 1. Tensor

### 1.1 Create Tensor

```py
torch.Tensor(*sizes)          # 基础构造函数（numpy可快速转换，共享内存）
torch.tensor(data,)           # 类似np.array的构造函数（拷贝，不共享内存）
torch.ones(*sizes)            # 全1Tensor
torch.zeros(*sizes)           # 全0Tensor
torch.eye(sizes)              # 对角线为1，其他为0
torch.arange(s, e, step)      # 从s到e，步长为step
torch.linspace(s, e, steps)   # 从s到e，均匀切分成steps份
torch.rand(*sizes)            # 均匀分布
torch.randn(*sizes)           # 标准分布
torch.normal(mean, std)       # 正态分布
torch.uniform(from, to)       # 均匀分布
torch.randperm(m)             # 随机排列
```

Tips: 将list转换为Tensor推荐使用`torch.tensor()`方法（0.4.0后支持）

### 1.2 Tensor Operation

#### 1) 形状操作

```py
tensor.view(*sizes)     # 调整tensor的形状，要保证前后元素总数一致，共享内存
tensor.unsqueeze(dim)   # 升维，在dim维（下标从0开始）上增加'1'
tensor.squeeze(dim)     # 降维，把dim维的'1'压缩掉，不传参数时将所有维度为'1'的压缩
tensor.resize(*sizes)   # 修改size，要保证前后元素总数一致，当使用resize_()时，如果大小超过原大小，会自动分配新的内存空间（新分配的值取决于内存状态）
tensor.item()           # 获取原始python对象数值（只对包含一个元素的tensor适用）
```

#### 2) 逐元素操作

```py
torch.abs(tensor)   # abs/sqrt/div/exp/fmod/log/pow.. 绝对值/平方根/除法/指数/求余/求幂..
torch.cos(tensor)   # cos/sin/asin/atan2/cosh.. 相关三角函数
torch.ceil(tensor)  # ceil/round/floor/trunc 上取整/四舍五入/下取整/只保留整数部分
torch.clamp(tensor, min, max) # 超过min喝和max部分截断（取到最大值或最小值）
torch.sigmoid       # sigmod/tanh.. 激活函数
```

#### 3) 归并操作

大部分函数都有一个参数`dim`，用来指定在哪个维度上执行操作。

假设输入的形状是(m, n, k)

- 如果指定dim=0，输出的形状就是(1, n, k)或者(n, k)
- 如果指定dim=1，输出的形状就是(m, 1, k)或者(m, k)
- 如果指定dim=2，输出的形状就是(m, n, 1)或者(m, n)

size中是否有"1"，取决于参数`keepdim`，`keepdim=True`会保留维度`1`。

```py
tensor.sum(dim, keepdim=True)   # 累加第dim维的数据，保留维数('1') mean/sum/median/mode 均值/和/中位数/众数
tensor.norm(num)                # 计算第num范数
tensor.dist(tensor_b, p=2)      # 计算两个tensor的距离，默认使用2范数
tensor.std(dim, unbiased=True, keepdim=False)   # 计算标准差， var计算方差
tensor.cumsum(dim)              # 累加，cumprod 累乘
```

#### 4) 比较操作

```py
torch.max(tensor, dim, keepdim)             # 取第dim维的最大值, min 最小值
torch.min(tensor_a, tensor_b)               # 取两个tensor中较小的元素
torch.topk(tensor, k, dim, largest, sorted) # 最大的k个数
```

#### 5) 线性代数

```py
tensor.trace()              # 计算tensor的迹
tensor.diag()               # 取对角线元素
tensor.t().contiguous()     # 转置，将存储空间变为连续
tensor.svd()                # 奇异值分解
tensor.mul(tensor_b)        # 点乘操作 等价于重载后的 '*' 运算符
tensor.matmul(tensor_b)     # 矩阵乘法 等价于重载后的 '@' 运算符 等价于 tensor.mm(tensor_b)
```

Tips: 
- 对于很多操作，例如gt、eq、div、mul、pow、fmod等，PyTorch都实现了运算符重载，所以可以直接使用运算符。如`a ** 2` 等价于`torch.pow(a,2)`, `a * 2`等价于`torch.mul(a,2)`
- 操作带下划线 `_` 的表示会修改tensor自身，不带下划线则操作返回修改后的Tensor

### 1.3 Tensor Index

```py
tensor[:2]          # 前两行
tensor[:, 2]        # 前两列
tensor[:2, 0:2]     # 前两行，第0, 1列
tensor[None]        # 等价于 tensor.unsqueeze(0)
tensor[:, None]     # 等价于 tensor.unsqueeze(1)
tensor > 1          # 返回一个 ByteTensor，结果和numpy一致
tensor[tensor > 1]  # 等价于 tensor.masked_select(tensor > 1) 选择结果与原tensor不共享内存空间
```

Tips: 高级索引(advanced indexing)和numpy是一样的

### 1.4 Tensor Type

```py
torch.set_default_tensor_type(torch.DoubleTensor) # 设置浮点数Tensor的类型，只能是half/float/double
torch.set_default_dtype(torch.float64)            # 两种函数效果一样，传参不一样
tensor.dtype        # 查看tensor类型
tensor.float()      # torch.FloatTensor
tensor.double()     # torch.DoubleTensor
tensor.long()       # torch.LongTensor
tensor.int()        # torch.IntTensor
tensor.type(torch.FloatTensor)  # 等价于 tensor.float()
tensor.type_as(tensor_b)        # 等价于 self.type(tensor_b.type())
```

Tips: 使用 `torch.set_default_tensor_type()` 或 `torch.set_default_dtype()` 后只会改变浮点数的类型，像 `torch.arange()` 这种返回整数型Tensor的函数不会改变自身的数据类型。

| Data type                | dtype                             | CPU tensor                                                   | GPU tensor                |
| ------------------------ | --------------------------------- | ------------------------------------------------------------ | ------------------------- |
| 32-bit floating point    | `torch.float32` or `torch.float`  | `torch.FloatTensor`                                          | `torch.cuda.FloatTensor`  |
| 64-bit floating point    | `torch.float64` or `torch.double` | `torch.DoubleTensor`                                         | `torch.cuda.DoubleTensor` |
| 16-bit floating point    | `torch.float16` or `torch.half`   | `torch.HalfTensor`                                           | `torch.cuda.HalfTensor`   |
| 8-bit integer (unsigned) | `torch.uint8`                     | [`torch.ByteTensor`](https://pytorch.org/docs/stable/tensors.html#torch.ByteTensor) | `torch.cuda.ByteTensor`   |
| 8-bit integer (signed)   | `torch.int8`                      | `torch.CharTensor`                                           | `torch.cuda.CharTensor`   |
| 16-bit integer (signed)  | `torch.int16` or `torch.short`    | `torch.ShortTensor`                                          | `torch.cuda.ShortTensor`  |
| 32-bit integer (signed)  | `torch.int32` or `torch.int`      | `torch.IntTensor`                                            | `torch.cuda.IntTensor`    |
| 64-bit integer (signed)  | `torch.int64` or `torch.long`     | `torch.LongTensor`                                           | `torch.cuda.LongTensor`   |

```py
# 以下操作都会继承 tensor 的 dtype/device/layer
torch.zeros_like(tensor)    # 等价于 torch.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device)
torch.rand_like(tensor)     # 等价于 torch.rand(tensor.size(), dtype=input.dtype, device=input.device)
tensor.new_ones(*sizes)     # 等价于 torch.ones(*sizes, dtyde=tensor.dtype, device=tensor.device)
tensor.new_tensor(data)     # 等价于 torch.tensor(data, dtype=tensor.dtype, device=tensor.device)
```

### 1.5 Tensor Memory

> 默认的tensor是FloatTensor，可通过`torch.set_default_tensor_type` 来修改默认tensor类型(如果默认类型为GPU tensor，则所有操作都将在GPU上进行)。Tensor的类型对分析内存占用很有帮助。例如对于一个size为(1000, 1000, 1000)的FloatTensor，它有`1000*1000*1000=10^9`个元素，每个元素占32bit/8 = 4Byte内存，所以共占大约4GB内存/显存。HalfTensor是专门为GPU版本设计的，同样的元素个数，显存占用只有FloatTensor的一半，所以可以极大缓解GPU显存不足的问题，但由于HalfTensor所能表示的数值大小和精度有限[^2]，所以可能出现溢出等问题。

[^2]: https://stackoverflow.com/questions/872544/what-range-of-numbers-can-be-represented-in-a-16-32-and-64-bit-ieee-754-syste

## a. 辅助工具

#### 1) 将Tensor转为图像

```py
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

show = ToPILImage()
# one image
show(image).resize((100, 100))

# multiple images
show(make_grid(images)).resize((400, 100))
```

#### 2) 查看性能

```py
def for_loop_add(x, y):
    result = []
    for i,j in zip(x, y):
        result.append(i + j)
    return torch.Tensor(result)

x = torch.zeros(100)
y = torch.ones(100)
%timeit -n 10 for_loop_add(x, y)
%timeit -n 10 x + y
```
```
777 µs ± 17 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
The slowest run took 10.00 times longer than the fastest. This could mean that an intermediate result is being cached.
6.2 µs ± 8.48 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

## b. CheatSheet

#### 1) 手撕线性回归

```py
import torch
torch.set_default_dtype(torch.float32)
%matplotlib inline
import matplotlib.pyplot as plt
from IPython import display

device = torch.device('cuda:0')

# 设置随机数种子，保证在不同电脑上运行时下面的输出一致
torch.manual_seed(1000) 

def get_fake_data(batch_size=8):
    ''' 产生随机数据：y=x*2+3，加上了一些噪声'''
    x = torch.rand(batch_size, 1, device=device) * 5
    y = x * 2 + 3 +  torch.randn(batch_size, 1, device=device)
    return x, y

# 随机初始化参数
w = torch.rand(1, 1).to(device)
b = torch.zeros(1, 1).to(device)

lr = 0.02 # 学习率

for ii in range(500):
    x, y = get_fake_data(batch_size=4)
    # forward：计算loss
    y_pred = x * w + b # x@W等价于x.mm(w);for python3 only
    loss = 0.5 * (y_pred - y) ** 2 # 均方误差
    loss = loss.mean()
    
    # backward：手动计算梯度
    dloss = 1
    dy_pred = dloss * (y_pred - y)
    
    dw = x.t() @ (dy_pred)
    db = dy_pred.sum()
    
    # 更新参数
    w.sub_(lr * dw)
    b.sub_(lr * db)
    
    if ii%50 ==0:
        # 画图
        display.clear_output(wait=True)
        x = torch.arange(0, 6, device=device, dtype=torch.float32).view(-1, 1)
        y = x @ w + b
        plt.plot(x.cpu().numpy(), y.cpu().numpy()) # predicted
        
        x2, y2 = get_fake_data(batch_size=32) 
        plt.scatter(x2.cpu().numpy(), y2.cpu().numpy()) # true data
        
        plt.xlim(0, 5)
        plt.ylim(0, 13)
        plt.show()
        plt.pause(0.5)
        
print('w: ', w.item(), 'b: ', b.item())
```

## Reference

> pytorch-book: https://github.com/chenyuntc/pytorch-book