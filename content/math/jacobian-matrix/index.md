---
title: "雅可比矩阵"
date: 2026-07-23
lastmod: 2026-07-23
draft: false
description: "从单变量导数和梯度出发理解雅可比矩阵，结合局部线性化、神经网络与反向传播说明它的计算方法和核心作用。"
summary: "雅可比矩阵把多输入、多输出函数的一阶偏导数组织成矩阵，是高维函数的局部线性映射，也是理解神经网络链式法则与反向传播的关键。"
tags: ["数学知识", "雅可比矩阵", "微积分", "线性代数", "反向传播"]
categories: ["数学知识"]
math: true
ShowToc: true
TocOpen: true
---

雅可比矩阵可以看作导数在高维空间中的推广。单变量函数只有一个输入方向和一个输出方向，因此一个导数就足以描述局部变化；当函数同时拥有多个输入和多个输出时，就需要用一个矩阵记录所有输出相对于所有输入的变化率。

## 雅可比矩阵是什么

### 从单变量函数的导数说起

对于单变量函数

$$
y=f(x),
$$

导数为

$$
f'(x)=\frac{\mathrm{d}y}{\mathrm{d}x}.
$$

它描述了输入 $x$ 发生微小变化时，输出 $y$ 会以多快的速度变化。例如

$$
y=x^2,\qquad \frac{\mathrm{d}y}{\mathrm{d}x}=2x.
$$

当 $x=3$ 且 $\Delta x=0.01$ 时，可以利用导数估计输出的变化：

$$
\Delta y\approx 2x\Delta x
=6\times 0.01
=0.06.
$$

这里的导数就是一维函数在当前点附近的“斜率”。

### 多个输入、一个输出：梯度

如果函数有多个输入，但只有一个标量输出：

$$
y=f(x_1,x_2,\ldots,x_n),
$$

那么需要分别计算 $y$ 对每个输入的偏导数，并把它们组成梯度：

$$
\nabla_{\boldsymbol{x}}y=
\begin{bmatrix}
\dfrac{\partial y}{\partial x_1}\\
\dfrac{\partial y}{\partial x_2}\\
\vdots\\
\dfrac{\partial y}{\partial x_n}
\end{bmatrix}.
$$

例如

$$
y=x_1^2+3x_2,
$$

其梯度为

$$
\nabla_{\boldsymbol{x}}y=
\begin{bmatrix}
2x_1\\
3
\end{bmatrix}.
$$

梯度记录的是：一个标量输出对每个输入方向分别有多敏感。

### 多个输入、多个输出：雅可比矩阵

进一步考虑多输入、多输出函数：

$$
\boldsymbol{x}=
\begin{bmatrix}
x_1\\
\vdots\\
x_n
\end{bmatrix},
\qquad
\boldsymbol{y}=f(\boldsymbol{x})=
\begin{bmatrix}
y_1\\
\vdots\\
y_m
\end{bmatrix}.
$$

每个输出 $y_i$ 都可能依赖每个输入 $x_j$。把全部一阶偏导数按照“输出为行、输入为列”排列，就得到雅可比矩阵：

$$
J_f(\boldsymbol{x})
=\frac{\partial\boldsymbol{y}}{\partial\boldsymbol{x}}
=\begin{bmatrix}
\dfrac{\partial y_1}{\partial x_1} &
\dfrac{\partial y_1}{\partial x_2} &
\cdots &
\dfrac{\partial y_1}{\partial x_n}\\
\dfrac{\partial y_2}{\partial x_1} &
\dfrac{\partial y_2}{\partial x_2} &
\cdots &
\dfrac{\partial y_2}{\partial x_n}\\
\vdots & \vdots & \ddots & \vdots\\
\dfrac{\partial y_m}{\partial x_1} &
\dfrac{\partial y_m}{\partial x_2} &
\cdots &
\dfrac{\partial y_m}{\partial x_n}
\end{bmatrix}
\in\mathbb{R}^{m\times n}.
$$

在本文采用的约定中，

$$
\left[J_f(\boldsymbol{x})\right]_{ij}
=\frac{\partial y_i}{\partial x_j}.
$$

因此：

- 第 $i$ 行是输出 $y_i$ 对全部输入的梯度；
- 第 $j$ 列表示输入 $x_j$ 对全部输出的影响；
- 行数等于输出维度，列数等于输入维度。

有些教材会采用转置后的排列方式，因此阅读公式时应先确认作者对雅可比矩阵形状的约定。

## 具体计算一个雅可比矩阵

考虑函数

$$
y_1=x_1^2+3x_2,\qquad y_2=x_1x_2.
$$

写成向量形式：

$$
f\!\left(
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
\right)
=\begin{bmatrix}
x_1^2+3x_2\\
x_1x_2
\end{bmatrix}.
$$

分别计算每个输出对每个输入的偏导数：

$$
\frac{\partial y_1}{\partial x_1}=2x_1,\qquad
\frac{\partial y_1}{\partial x_2}=3,
$$

$$
\frac{\partial y_2}{\partial x_1}=x_2,\qquad
\frac{\partial y_2}{\partial x_2}=x_1.
$$

所以

$$
J_f(\boldsymbol{x})
=\begin{bmatrix}
2x_1 & 3\\
x_2 & x_1
\end{bmatrix}.
$$

在点

$$
\boldsymbol{x}=
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

处，雅可比矩阵为

$$
J_f(1,2)=
\begin{bmatrix}
2 & 3\\
2 & 1
\end{bmatrix}.
$$

这个矩阵给出了函数在 $(1,2)$ 附近的局部变化关系：第一列表示 $x_1$ 的微小变化会分别以约 $2$ 倍的速度影响 $y_1$ 和 $y_2$；第二列表示 $x_2$ 的微小变化会分别以约 $3$ 倍和 $1$ 倍的速度影响两个输出。

## 雅可比矩阵最核心的含义：局部线性化

只会计算偏导数，还没有真正理解雅可比矩阵。它最核心的作用，是把一个光滑的非线性函数在某一点附近近似成线性映射：

$$
f(\boldsymbol{x}+\Delta\boldsymbol{x})
\approx
f(\boldsymbol{x})+J_f(\boldsymbol{x})\Delta\boldsymbol{x}.
$$

也可以写成

$$
\Delta\boldsymbol{y}
\approx
J_f(\boldsymbol{x})\Delta\boldsymbol{x}.
$$

继续使用上一节的例子。在 $(1,2)$ 附近，令

$$
\Delta\boldsymbol{x}=
\begin{bmatrix}
0.01\\
-0.02
\end{bmatrix},
$$

则

$$
\Delta\boldsymbol{y}
\approx
\begin{bmatrix}
2 & 3\\
2 & 1
\end{bmatrix}
\begin{bmatrix}
0.01\\
-0.02
\end{bmatrix}
=\begin{bmatrix}
-0.04\\
0
\end{bmatrix}.
$$

这意味着输入按上述方向发生微小变化时，$y_1$ 预计减少约 $0.04$，而 $y_2$ 在一阶近似下保持不变。

单变量导数是一个数，因为输入和输出都只有一个方向；高维函数有许多输入方向和输出方向，因此它的“斜率”自然表现为一个矩阵。雅可比矩阵就是这个高维斜率。

## 雅可比矩阵与神经网络

神经网络本质上是一个由参数控制的多输入、多输出函数：

$$
\boldsymbol{y}=f(\boldsymbol{x};\boldsymbol{\theta}),
$$

其中 $\boldsymbol{x}$ 是输入，$\boldsymbol{y}$ 是模型输出，$\boldsymbol{\theta}$ 是模型参数。因此至少有两类重要的雅可比矩阵：

$$
J_{\boldsymbol{x}}
=\frac{\partial\boldsymbol{y}}{\partial\boldsymbol{x}},
\qquad
J_{\boldsymbol{\theta}}
=\frac{\partial\boldsymbol{y}}{\partial\boldsymbol{\theta}}.
$$

前者描述输出对输入的敏感程度，后者描述输出对模型参数的敏感程度。

### 线性层的雅可比矩阵

对于线性层

$$
\boldsymbol{y}=W\boldsymbol{x}+\boldsymbol{b},
\qquad
W\in\mathbb{R}^{m\times n},
$$

第 $i$ 个输出可以写成

$$
y_i=\sum_j W_{ij}x_j+b_i.
$$

于是

$$
\frac{\partial y_i}{\partial x_j}=W_{ij},
$$

从而得到

$$
\frac{\partial\boldsymbol{y}}{\partial\boldsymbol{x}}=W.
$$

也就是说，线性层相对于输入的雅可比矩阵就是它的权重矩阵。

### 激活函数的雅可比矩阵

假设激活函数逐元素作用：

$$
\boldsymbol{h}=\sigma(\boldsymbol{z}),
\qquad
h_i=\sigma(z_i).
$$

因为 $h_i$ 只依赖 $z_i$，所以

$$
\frac{\partial h_i}{\partial z_j}
=\begin{cases}
\sigma'(z_i), & i=j,\\
0, & i\ne j.
\end{cases}
$$

它的雅可比矩阵是对角矩阵：

$$
J_\sigma(\boldsymbol{z})
=\operatorname{diag}\!\left(
\sigma'(z_1),\sigma'(z_2),\ldots,\sigma'(z_n)
\right).
$$

以 ReLU 为例：

$$
\operatorname{ReLU}(z)=\max(0,z),
$$

在 $z\ne 0$ 时，

$$
\operatorname{ReLU}'(z)
=\begin{cases}
1, & z>0,\\
0, & z<0.
\end{cases}
$$

在 $z=0$ 处函数不可导，深度学习框架通常选取一个约定值。ReLU 的雅可比矩阵在对角线上主要由 $0$ 和 $1$ 组成：正值位置允许梯度通过，负值位置会把梯度截断。

### 两层神经网络的雅可比矩阵

考虑一个两层网络：

$$
\boldsymbol{a}=W_1\boldsymbol{x}+\boldsymbol{b}_1,
$$

$$
\boldsymbol{h}=\sigma(\boldsymbol{a}),
$$

$$
\boldsymbol{y}=W_2\boldsymbol{h}+\boldsymbol{b}_2.
$$

根据链式法则，输出对输入的雅可比矩阵为

$$
\frac{\partial\boldsymbol{y}}{\partial\boldsymbol{x}}
=W_2\operatorname{diag}\!\left(\sigma'(\boldsymbol{a})\right)W_1.
$$

这就是链式法则的矩阵形式。网络每经过一层，整体雅可比矩阵就要与该层的局部雅可比矩阵相乘。

## 雅可比矩阵和反向传播

雅可比矩阵在 AI 中最核心的应用之一就是反向传播。设神经网络为

$$
\boldsymbol{y}=f(\boldsymbol{x}),
$$

损失函数为

$$
L=\ell(\boldsymbol{y}),
$$

其中 $\boldsymbol{y}$ 是向量，$L$ 是标量。为了计算损失对输入的梯度，根据链式法则有

$$
\nabla_{\boldsymbol{x}}L
=J_f(\boldsymbol{x})^{\mathsf{T}}\nabla_{\boldsymbol{y}}L,
$$

其中

$$
J_f(\boldsymbol{x})
=\frac{\partial\boldsymbol{y}}{\partial\boldsymbol{x}}.
$$

如果 $\boldsymbol{x}\in\mathbb{R}^n$、$\boldsymbol{y}\in\mathbb{R}^m$，那么

$$
J_f\in\mathbb{R}^{m\times n},
\qquad
\nabla_{\boldsymbol{y}}L\in\mathbb{R}^m,
$$

因此

$$
J_f^{\mathsf{T}}\nabla_{\boldsymbol{y}}L
\in\mathbb{R}^n,
$$

恰好得到损失相对于输入的梯度。类似地，损失对模型参数的梯度为

$$
\nabla_{\boldsymbol{\theta}}L
=J_{\boldsymbol{\theta}}^{\mathsf{T}}\nabla_{\boldsymbol{y}}L,
\qquad
J_{\boldsymbol{\theta}}
=\frac{\partial\boldsymbol{y}}{\partial\boldsymbol{\theta}}.
$$

因此，反向传播的本质可以概括为：

> 从输出端开始，把下游梯度依次乘以各层局部雅可比矩阵的转置，使梯度沿计算图反向传递。

实际的自动微分系统通常不会显式构造完整雅可比矩阵，因为它可能非常庞大。反向传播直接计算“转置雅可比矩阵与向量的乘积”，也就是向量—雅可比积，从而在保留链式法则结果的同时显著节省内存和计算量。

## 总结

雅可比矩阵把多输入、多输出函数的全部一阶偏导数组织为一个矩阵：

$$
\left[J_f(\boldsymbol{x})\right]_{ij}
=\frac{\partial y_i}{\partial x_j}.
$$

它有三个层层递进的理解角度：

1. **计算角度**：每一行是一个输出对全部输入的梯度；
2. **几何角度**：它是非线性函数在某一点附近的局部线性映射；
3. **神经网络角度**：各层雅可比矩阵通过链式法则相乘，反向传播则不断计算转置雅可比矩阵与下游梯度的乘积。

从一维导数到梯度，再到雅可比矩阵，本质上都是在回答同一个问题：输入发生微小变化时，输出会怎样变化。

