# pytorch-cheatsheet

## 1. 数据预处理

#### 1) 将一维数组的数据转为一维图像

```py
# 例如每行数据转为 6x6 的图片
# inputs: [[0., 1.2, 3.2, 4.5], [1., 2.1, 3., 4.2] ...]
image_size = 6

inputs_img = np.pad(inputs, ((0, 0), (0, image_size ** 2 - len(inputs[0]))), 'constant').reshape(-1, 1, image_size, image_size)
```

#### 2) 将数据集拆解为训练集和验证集（有标签）

```py
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.3)
```

## 2. GPU

#### 1) 单GPU的使用

一般来说需要转移到GPU的就是model和data了（注意optimizer要在model转移到GPU后定义）

> PyTorch 0.4.0 后推荐使用 `to(device)` 方法来将 Tensor 和 Modules 移动到不同设备, 代替以前的 `cpu()` 或 `cuda()`

```py
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

inputs = inputs.to(device)
model = model.to(device)

result = model(inputs).to('cpu')
print(result)
```

#### 2) 多GPU的使用

```py
# TODO
```
