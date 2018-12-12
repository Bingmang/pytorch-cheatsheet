# pytorch-cheatsheet

## 1. Tensor

### 1.1 Create Tensor

```py
torch.Tensor(*sizes)          # 基础构造函数
torch.tensor(data,)           # 类似np.array的构造函数
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

```py
tensor.view(*sizes)     # 调整tensor的形状，要保证前后元素总数一致，共享内存
tensor.unsqueeze(dim)   # 升维，在dim维（下标从0开始）上增加'1'
tensor.squeeze(dim)     # 降维，把dim维的'1'压缩掉，不传参数时将所有维度为'1'的压缩
tensor.resize(*sizes)   # 修改size，要保证前后元素总数一致，当使用resize_()时，如果大小超过原大小，会自动分配新的内存空间（新分配的值取决于内存状态）
tensor.item()           # 获取原始python对象数值（只对包含一个元素的tensor适用）
```

Tips: 操作带下划线 `_` 的表示会修改tensor自身，不带下划线则操作返回修改后的Tensor

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

### 1.4 Tensor Type Convert

```py
torch.set_default_tensor_type('torch.DoubleTensor') # 设置默认tensor，传入的是字符串
tensor.dtype        # 查看tensor类型
tensor.float()      # torch.FloatTensor
tensor.double()     # torch.DoubleTensor
tensor.long()       # torch.LongTensor
tensor.int()        # torch.IntTensor
```

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

## Reference

> pytorch-book: https://github.com/chenyuntc/pytorch-book