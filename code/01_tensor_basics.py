import torch

x = torch.arange(12)
print(x)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

print(x.shape)
# torch.Size([12])

print(x.numel())
# 12 元素的数量，永远是标量

X = x.reshape(3, 4)
print(X)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]]) 改变这个数组为3行四列

print(torch.ones(2, 3, 4))
# tensor([[[1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.]],

#         [[1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.]]]) 全1数组，3行4列，两层

print(torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]]) 手动创建一个3行3列的数组

x = torch.tensor([1.0, 2, 3, 4])
y = torch.tensor([4, 3, 2, 1])
print(x + y, x - y, x * y, x / y, x ** y)
# tensor([5., 5., 5., 5.]) tensor([-3., -1.,  1.,  3.]) tensor([4., 6., 6., 4.]) tensor([0.2500, 0.6667, 1.5000, 4.0000]) tensor([1., 8., 9., 4.])
# 按元素做运算

x = torch.arange(12, dtype = torch.float32).reshape(3, 4)
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((x, y), dim = 0), torch.cat((x, y), dim = 1))
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [ 2.,  1.,  4.,  3.],
#         [ 1.,  2.,  3.,  4.],
#         [ 4.,  3.,  2.,  1.]]) tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
#         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
#         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]) 直接将矩阵合并，dim=0表示按行合并，dim=1表示按列合并

print(x == y)
# tensor([[False,  True, False,  True],
#         [False, False, False, False],
#         [False, False, False, False]]) 通过运算符可以构建一个二元张量

print(x.sum())
# tensor(66.) 对张量内所有元素进行求和

a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print(a + b)
# tensor([[0, 1],
#         [1, 2],
#         [2, 3]]) 广播机制，如果两个张量维度相同，但是形状不一致，那么就会复制补全后再运算

A = torch.arange(20).reshape(5, 4)
print(A, "\n", A.T)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11],
#         [12, 13, 14, 15],
#         [16, 17, 18, 19]]) 
#  tensor([[ 0,  4,  8, 12, 16],
#         [ 1,  5,  9, 13, 17],
#         [ 2,  6, 10, 14, 18],
#         [ 3,  7, 11, 15, 19]]) 矩阵的转置

B = A.clone()
print(A * B)
# tensor([[  0,   1,   4,   9],
#         [ 16,  25,  36,  49],
#         [ 64,  81, 100, 121],
#         [144, 169, 196, 225],
#         [256, 289, 324, 361]]) 两个矩阵按元素进行乘法叫作哈达玛积

A = torch.arange(18).reshape(2, 3, 3)
A_sum_axis0 = A.sum(axis = 0)
A_sum_axis1 = A.sum(axis = 1)
A_sum_axis2 = A.sum(axis = 2)
print(A,"\n", A_sum_axis0, "\n", A_sum_axis1, "\n", A_sum_axis2)
# tensor([[[ 0,  1,  2],
#          [ 3,  4,  5],
#          [ 6,  7,  8]],

#         [[ 9, 10, 11],
#          [12, 13, 14],
#          [15, 16, 17]]]) 
#  tensor([[ 9, 11, 13],
#         [15, 17, 19],
#         [21, 23, 25]]) 
#  tensor([[ 9, 12, 15],
#         [36, 39, 42]]) 
#  tensor([[ 3, 12, 21],
#         [30, 39, 48]]) 指定求和汇总张量的轴，按照哪个轴求和，哪个轴就归为1（层，行，列）

print(A.sum(), "\n", A.numel(), "\n", A.float().mean())
# tensor(153) 
#  18 
#  tensor(8.5000) 求平均值就是把张量中的每个元素相加，再除以张量中元素的个数

print(A.cumsum(axis = 0))
# tensor([[[ 0,  1,  2],
#          [ 3,  4,  5],
#          [ 6,  7,  8]],

#         [[ 9, 11, 13],
#          [15, 17, 19],
#          [21, 23, 25]]]) 对于指定张量轴进行累加求和

A = torch.arange(6).reshape(2, 3)
x = torch.arange(4, 7)
print(A, x, torch.mv(A, x))
# tensor([[0, 1, 2],
#         [3, 4, 5]]) tensor([4, 5, 6]) tensor([17, 62]) 矩阵点乘向量是mv

B = torch.arange(6).reshape(3, 2)
print(A, "\n", B, "\n", torch.mm(A, B))
# tensor([[0, 1, 2],
#         [3, 4, 5]]) 
#  tensor([[0, 1],
#         [2, 3],
#         [4, 5]]) 
#  tensor([[10, 13],
#         [28, 40]]) 使用mm实现矩阵乘法

u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
# tensor(5.) 使用norm来求L2范数

u = torch.arange(4, dtype = torch.float32).reshape(2, 2)
print(torch.norm(u))
# tensor(3.7417) 矩阵的f范数原理和向量是一样的，都是平方和的平方根