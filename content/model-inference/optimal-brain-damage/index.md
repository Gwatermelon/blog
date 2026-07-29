---
title: "Optimal Brain Damage（OBD）：用 Hessian 判断权重重要性"
date: 2026-07-29
lastmod: 2026-07-29
draft: false
description: "从 Hessian 矩阵和二阶泰勒展开出发，推导 Optimal Brain Damage 如何估计删除单个权重造成的损失增量，并解释它为何优于单纯的幅值剪枝。"
summary: "权重绝对值小不等于权重不重要。OBD 使用 Hessian 对角元素衡量损失对各权重方向的曲率，以二阶显著性分数选择删除后损失增加最小的权重。"
tags: ["模型推理", "LLM Inference", "模型剪枝", "Optimal Brain Damage", "Hessian", "模型压缩"]
categories: ["模型推理"]
math: true
ShowToc: true
TocOpen: true
---

> **核心思想：** OBD 不只看权重有多小，而是利用 Hessian 对角元素判断损失函数对该权重方向有多敏感，进而选择删除后损失增量最小的权重。

Optimal Brain Damage（OBD）要回答的问题是：**当必须删除一些权重时，怎样判断删除哪个权重对模型损失的影响最小？**

OBD 由 Yann LeCun、John S. Denker 和 Sara A. Solla 提出。论文收录于 *Advances in Neural Information Processing Systems 2（NIPS 1989）*；一些参考文献也会按照论文集出版年份将其标为 1990。

理解 OBD 之前，需要先理解 Hessian 矩阵及其对角元素表示的含义。

## 什么是 Hessian 矩阵

Hessian（海森）矩阵是一个由多元函数全部二阶偏导数组成的方阵。

考虑二元函数：

$$
f(x,y).
$$

它有两个输入变量，因此不仅需要考虑函数沿 $x$ 方向和 $y$ 方向的曲率，还需要考虑两个变量之间的相互影响。对应的二阶偏导数包括：

$$
\frac{\partial^2 f}{\partial x^2},
\qquad
\frac{\partial^2 f}{\partial x\partial y},
\qquad
\frac{\partial^2 f}{\partial y\partial x},
\qquad
\frac{\partial^2 f}{\partial y^2}.
$$

将它们排列成矩阵：

$$
H_f(x,y)=
\begin{bmatrix}
\dfrac{\partial^2 f}{\partial x^2}
&
\dfrac{\partial^2 f}{\partial x\partial y}
\\[6pt]
\dfrac{\partial^2 f}{\partial y\partial x}
&
\dfrac{\partial^2 f}{\partial y^2}
\end{bmatrix}.
$$

这就是函数 $f$ 的 Hessian 矩阵。

### 对角元素和非对角元素

假设：

$$
H=
\begin{bmatrix}
H_{11} & H_{12}\\
H_{21} & H_{22}
\end{bmatrix}.
$$

其中：

$$
H_{11}=\frac{\partial^2 f}{\partial x^2}
$$

表示函数沿 $x$ 方向的弯曲程度；

$$
H_{22}=\frac{\partial^2 f}{\partial y^2}
$$

表示函数沿 $y$ 方向的弯曲程度；而

$$
H_{12}=\frac{\partial^2 f}{\partial x\partial y}
$$

表示变量之间的耦合关系：当 $y$ 发生变化时，函数关于 $x$ 的斜率会怎样变化。

因此：

- 对角元素 $H_{ii}$ 描述损失函数沿单个参数方向的曲率或敏感度；
- 非对角元素 $H_{ij}$ 描述参数 $w_i$ 与 $w_j$ 之间的耦合关系。

### Hessian 的计算示例

考虑函数：

$$
f(x,y)=x^2+xy+3y^2.
$$

先求一阶偏导：

$$
\frac{\partial f}{\partial x}=2x+y,
\qquad
\frac{\partial f}{\partial y}=x+6y.
$$

因此梯度为：

$$
\nabla f(x,y)=
\begin{bmatrix}
2x+y\\
x+6y
\end{bmatrix}.
$$

再对梯度求一次偏导。对于梯度的第一个分量 $2x+y$：

$$
\frac{\partial(2x+y)}{\partial x}=2,
\qquad
\frac{\partial(2x+y)}{\partial y}=1.
$$

对于梯度的第二个分量 $x+6y$：

$$
\frac{\partial(x+6y)}{\partial x}=1,
\qquad
\frac{\partial(x+6y)}{\partial y}=6.
$$

所以 Hessian 矩阵为：

$$
H_f=
\begin{bmatrix}
2 & 1\\
1 & 6
\end{bmatrix}.
$$

这个例子中的 Hessian 不依赖 $x$、$y$，因为原函数是二次函数。

## OBD：Optimal Brain Damage

### OBD 要解决什么问题

一个训练好的神经网络通常包含大量权重。为了减小模型规模和计算量，我们希望将一些不重要的权重设置为零，也就是进行权重剪枝。

假设模型中有一个权重向量：

$$
\mathbf{w}=[w_1,w_2,\ldots,w_q,\ldots,w_n].
$$

如果想要删除第 $q$ 个权重，可以把它设置为零：

$$
w_q'=0.
$$

一个线性层原本可能计算：

$$
y=w_1x_1+w_2x_2+\cdots+w_qx_q+\cdots+w_nx_n.
$$

将 $w_q$ 设置为零后：

$$
y'=w_1x_1+w_2x_2+\cdots+0\cdot x_q+\cdots+w_nx_n.
$$

于是 $w_qx_q$ 这条连接不再对计算结果产生影响，等价于将这条神经网络连接剪掉。

参数变化定义为：

$$
\delta w_q=w_q'-w_q.
$$

删除权重意味着：

$$
w_q'=0.
$$

所以：

$$
\delta w_q=0-w_q=-w_q.
$$

### 最直接的方法：幅值剪枝

最简单的剪枝策略是删除绝对值最小的权重。

假设模型有三个权重：

$$
\mathbf{w}=[2.0,\ 0.01,\ -1.5].
$$

如果必须删除一个权重，最直观的选择是：

$$
w_2=0.01,
$$

因为它离零最近。删除不同权重时，参数变化量分别为：

$$
\begin{aligned}
|\delta w_1|&=|0-2.0|=2.0,\\
|\delta w_2|&=|0-0.01|=0.01,\\
|\delta w_3|&=|0-(-1.5)|=1.5.
\end{aligned}
$$

删除 $w_2$ 对参数的改动最小。因此，幅值剪枝使用的基本直觉是：

> 权重绝对值越小，将它设置为零造成的变化越小，对模型的影响也可能越小。

### 幅值剪枝的问题

“权重小，所以不重要”只是一个非常粗糙的假设。一个权重对模型的影响，不仅取决于权重本身，还取决于：

- 对应的输入激活有多大；
- 损失函数对该权重有多敏感；
- 其他权重能否补偿它；
- 该权重与其他参数之间是否存在关联。

例如有一个线性计算：

$$
y=w_1x_1+w_2x_2.
$$

假设：

$$
w_1=0.01,\qquad x_1=1000,
$$

$$
w_2=1,\qquad x_2=0.001.
$$

两个连接的实际贡献分别是：

$$
w_1x_1=0.01\times1000=10,
$$

$$
w_2x_2=1\times0.001=0.001.
$$

虽然：

$$
|w_1|\lt|w_2|,
$$

但是 $w_1$ 对输出的影响远大于 $w_2$。如果只根据权重绝对值剪枝，就会删除 $w_1$，造成很大的输出变化。

因此：

> 权重小不等于权重不重要。

### 从直接试删到二阶近似

假设要删除第 $q$ 个权重：

$$
w_q\longrightarrow 0,
$$

对应参数变化为：

$$
\delta w_q=-w_q.
$$

最简单的显著性指标是：

$$
\operatorname{importance}(w_q)=|w_q|.
$$

但权重小不代表不重要。某个权重虽然数值小，但损失函数可能在该方向非常陡峭；轻微改变它，就可能导致损失显著增加。

OBD 因而同时考虑两个因素：

1. 权重有多大；
2. 损失函数对这个权重有多敏感。

如果模型有 $n$ 个权重：

$$
w_1,w_2,\ldots,w_n,
$$

理论上可以依次尝试删除每一个权重。例如：

- 删除 $w_1$，损失可能增加 $0.5$；
- 删除 $w_2$，损失可能增加 $0.001$；
- 删除 $w_3$，损失可能增加 $2.0$。

理想选择是：

$$
q^*
=\arg\min_q
\left[
L(\mathbf{w}\ \text{删除}\ w_q)-L(\mathbf{w})
\right],
$$

也就是选择删除后损失增加最少的权重。

但是，如果真的对每个权重都执行一次“删除、重新运行整个验证集、计算模型损失、再恢复原权重”，计算成本会非常高。

OBD 的目标就是：

> 不实际逐个删除并运行模型，而是利用二阶泰勒展开和 Hessian，对删除每个权重造成的损失变化进行近似估计。

## 为什么训练良好的模型可以使用二阶近似

假设模型当前参数为：

$$
\mathbf{w},
$$

剪枝后参数变为：

$$
\mathbf{w}+\delta\mathbf{w}.
$$

我们想知道剪枝会使损失函数增加多少：

$$
\Delta L=L(\mathbf{w}+\delta\mathbf{w})-L(\mathbf{w}).
$$

在当前参数附近，对损失函数进行泰勒展开：

$$
\Delta L
\approx
\nabla L(\mathbf{w})^T\delta\mathbf{w}
+\frac{1}{2}\delta\mathbf{w}^TH\delta\mathbf{w}
+O(\|\delta\mathbf{w}\|^3).
$$

其中：

- 第一项是一阶项，描述当前位置的斜率；
- 第二项是二阶项，描述损失曲面的弯曲程度；
- 后面的三次及以上项描述更复杂的高阶形状。

当模型已经训练得较好，并位于局部最优点附近时：

$$
\nabla L(\mathbf{w})\approx 0.
$$

如果同时假设当前邻域内的损失函数可以用二次函数近似，高阶项可以忽略，于是：

$$
\Delta L
\approx
\frac{1}{2}\delta\mathbf{w}^TH\delta\mathbf{w}.
$$

## Hessian 的对角元素和非对角元素

假设只有两个参数：

$$
\mathbf{w}=
\begin{bmatrix}
w_1\\
w_2
\end{bmatrix},
\qquad
H=
\begin{bmatrix}
H_{11} & H_{12}\\
H_{21} & H_{22}
\end{bmatrix}.
$$

二阶损失变化为：

$$
\Delta L
\approx
\frac{1}{2}
\begin{bmatrix}
\delta w_1 & \delta w_2
\end{bmatrix}
\begin{bmatrix}
H_{11} & H_{12}\\
H_{21} & H_{22}
\end{bmatrix}
\begin{bmatrix}
\delta w_1\\
\delta w_2
\end{bmatrix}.
$$

在 Hessian 对称，即 $H_{12}=H_{21}$ 时，展开得到：

$$
\Delta L
\approx
\frac{1}{2}
\left(
H_{11}\delta w_1^2
+2H_{12}\delta w_1\delta w_2
+H_{22}\delta w_2^2
\right).
$$

其中：

- $H_{11}\delta w_1^2$ 表示只改变 $w_1$ 带来的曲率影响；
- $H_{22}\delta w_2^2$ 表示只改变 $w_2$ 带来的曲率影响；
- $2H_{12}\delta w_1\delta w_2$ 表示两个权重同时改变时产生的相互作用。

因此，对角元素 $H_{qq}$ 表示单个参数方向上的敏感度，非对角元素 $H_{ij}$ 则表示参数 $w_i$ 与 $w_j$ 之间的耦合关系。

## OBD 为什么只看 Hessian 对角元素

OBD 进一步只保留 Hessian 的对角元素。这不是说参数之间真的彼此独立，而是一种让二阶剪枝能够实际计算的近似。

### 原因一：完整 Hessian 太大

假设模型有 $N$ 个参数，Hessian 的形状是：

$$
N\times N,
$$

需要存储：

$$
N^2
$$

个元素。

例如，一个只有一亿参数的模型：

$$
N=10^8.
$$

Hessian 的元素数量为：

$$
10^{16}.
$$

即使使用 FP32，每个元素占 4 字节，也需要：

$$
4\times10^{16}\ \text{bytes},
$$

约为 $40\ \text{PB}$，显然无法直接存储。

而 Hessian 对角线只有 $N$ 个元素，存储成本从：

$$
O(N^2)
$$

降低为：

$$
O(N).
$$

OBD 出现在计算条件有限的时代，完整 Hessian 在当时以及现代大模型上都很难直接使用。

### 原因二：需要给每个权重一个独立评分

OBD 希望为每个权重计算一个显著性：

$$
S_q=\frac{1}{2}H_{qq}w_q^2,
$$

然后对所有权重排序：

$$
S_{q_1}\lt S_{q_2}\lt\cdots.
$$

优先删除显著性最小的权重。

如果保留 Hessian 的非对角元素，损失变化会包含：

$$
H_{ij}\delta w_i\delta w_j.
$$

这意味着一个权重是否重要，不再只取决于它自己，还取决于其他哪些权重也被删除。例如，单独删除 $w_1$ 可能安全，单独删除 $w_2$ 也可能安全，但同时删除它们却可能造成很大影响。这时很难为每个权重分配一个完全独立的分数。

使用对角近似后：

$$
\Delta L
\approx
\frac{1}{2}\sum_i H_{ii}\delta w_i^2.
$$

每个权重的贡献可以独立计算：

$$
S_i=\frac{1}{2}H_{ii}w_i^2.
$$

这使剪枝排序变得非常简单。

### 原因三：假设权重之间的相互作用较弱

OBD 做了近似：

$$
H_{ij}\approx 0,
\qquad i\ne j.
$$

也就是假设不同权重方向之间的耦合较弱。于是 Hessian 近似为：

$$
H\approx
\begin{bmatrix}
H_{11} & 0 & \cdots\\
0 & H_{22} & \cdots\\
\vdots & \vdots & \ddots
\end{bmatrix}.
$$

但这只是为了简化计算，并不意味着神经网络中的参数真的彼此独立。实际上，神经网络权重之间往往存在较强相关性，这也是后续二阶剪枝方法继续改进 OBD 的重要原因。

## OBD 的显著性公式

删除第 $q$ 个权重时：

$$
\delta w_q=-w_q.
$$

在模型已经训练得较好的情况下，使用二阶近似：

$$
\Delta L
\approx
\frac{1}{2}\delta\mathbf{w}^TH\delta\mathbf{w}.
$$

OBD 进一步忽略 Hessian 的非对角元素，只保留第 $q$ 个权重对应的对角项：

$$
\Delta L_q
\approx
\frac{1}{2}H_{qq}(\delta w_q)^2.
$$

因为：

$$
(\delta w_q)^2=(-w_q)^2=w_q^2,
$$

所以：

$$
\boxed{
\Delta L_q
\approx
\frac{1}{2}H_{qq}w_q^2
}.
$$

OBD 将：

$$
\boxed{
S_q=\frac{1}{2}H_{qq}w_q^2
}
$$

定义为权重 $w_q$ 的显著性。

这个公式不只看权重幅值 $w_q^2$，还看 Hessian 对角元素 $H_{qq}$。其中，$H_{qq}$ 表示损失函数在 $w_q$ 这个参数方向上的弯曲程度，也就是损失对该权重变化的敏感程度。

## 具体例子

假设有两个权重：

$$
w_1=0.1,
\qquad
w_2=1.0.
$$

只看绝对值时：

$$
|w_1|\lt|w_2|,
$$

因此幅值剪枝会删除 $w_1$。

但是，假设 Hessian 的对角元素为：

$$
H_{11}=1000,
\qquad
H_{22}=0.01.
$$

OBD 显著性分别为：

$$
S_1
=\frac{1}{2}\times1000\times0.1^2
=5,
$$

$$
S_2
=\frac{1}{2}\times0.01\times1^2
=0.005.
$$

虽然 $w_2$ 比 $w_1$ 大很多，但删除 $w_2$ 对损失的近似影响反而更小：

$$
S_2\lt S_1.
$$

所以 OBD 会删除：

$$
w_2,
$$

而不是幅值更小的 $w_1$。这说明 OBD 真正解决的是：

> 不要根据权重大小剪枝，而要根据预计的损失增量剪枝。

## OBD 的假设与边界

OBD 的推导依赖几个关键近似：

- **极值点近似**：模型已经训练到局部最优点附近，因此一阶梯度近似为零；
- **二次近似**：损失函数在当前参数附近可以由二阶泰勒展开近似，高阶项可以忽略；
- **对角近似**：忽略不同权重之间的 Hessian 非对角项，将各权重的损失贡献视为可以独立相加。

这些近似让 OBD 的计算从理论上不可承受的完整 Hessian，简化为每个权重一个独立分数。但它也带来局限：当模型尚未收敛、剪枝扰动很大，或参数之间存在明显耦合时，$\frac{1}{2}H_{qq}w_q^2$ 可能无法准确预测真实损失变化。

## 总结

OBD 的推理链条可以概括为：

```text
删除权重
  → 参数扰动 δw
  → 二阶泰勒展开估计损失变化
  → 忽略一阶项、高阶项和 Hessian 非对角项
  → 得到每个权重的独立显著性分数
```

最终评分公式是：

$$
S_q=\frac{1}{2}H_{qq}w_q^2.
$$

权重越大，或损失函数沿该权重方向的曲率越大，删除它造成的预计损失增量就越大。相比只看 $|w_q|$ 的幅值剪枝，OBD 将模型对参数变化的敏感度纳入了剪枝依据。

## 参考资料

- [Optimal Brain Damage（NeurIPS Proceedings）](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html)

