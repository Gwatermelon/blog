---
title: "Optimal Brain Surgeon（OBS）：带补偿的二阶剪枝方法"
date: 2026-08-10
lastmod: 2026-08-10
draft: false
description: "从删除单个权重后的补偿直觉出发，推导 Optimal Brain Surgeon 如何利用完整 Hessian 和约束优化同时回答删哪个权重、其他权重应该如何调整。"
summary: "OBS 相比 OBD 的关键区别在于补偿：删除某个权重以后，它允许其他权重沿着 Hessian 给出的二阶曲率方向共同调整，从而尽量降低剪枝带来的损失增量。"
tags: ["模型推理", "LLM Inference", "模型剪枝", "Optimal Brain Surgeon", "Hessian", "模型压缩"]
categories: ["模型推理"]
math: true
ShowToc: true
TocOpen: true
---

> **核心直觉：** OBS 不是把某个权重置零后就结束，而是在必须删除该权重的约束下，让其他权重一起调整，用最小的二阶损失增量完成剪枝。

Optimal Brain Surgeon（OBS）要解决的问题可以拆成两句话：

1. 如果必须剪掉一个权重，应该剪掉哪一个？
2. 剪掉以后，其他权重应该分别补偿多少，才能让损失增加最少？

这正是 OBS 和 Optimal Brain Damage（OBD）的核心差别。OBD 使用 Hessian 的对角近似，为每个权重独立打分；OBS 则保留完整 Hessian，显式利用权重之间的耦合关系。

## 从一个线性例子理解补偿

先看一个最简单的线性模型：

$$
y=w_1x_1+w_2x_2.
$$

假设当前权重为：

$$
w_1=2,\qquad w_2=3,
$$

并且某次输入满足：

$$
x_1=1,\qquad x_2=1.
$$

原始输出为：

$$
y=2\times 1+3\times 1=5.
$$

现在想剪掉 $w_1$，也就是强制：

$$
w_1:2\longrightarrow 0.
$$

如果其他权重完全不动，新的输出变为：

$$
y'=0\times 1+3\times 1=3.
$$

输出从 $5$ 变成 $3$，模型行为显然发生了变化。

OBS 会继续问一个问题：既然 $w_1$ 必须删除，能不能让 $w_2$ 帮忙补回来？例如把 $w_2$ 从 $3$ 调整到 $5$：

$$
y'=0\times 1+5\times 1=5.
$$

这样即使 $w_1$ 被删除，输出仍然保持不变。这个例子虽然极简，但它揭示了 OBS 的核心：**删除一个权重以后，允许其他权重做最优补偿**。

## 为什么其他权重可以补偿

神经网络中的特征通常不是完全独立的。若两个输入特征高度相关，例如：

$$
x_1\approx x_2,
$$

那么：

$$
w_1x_1+w_2x_2
$$

本身就存在一定冗余。极端情况下，如果：

$$
x_1=x_2=x,
$$

则：

$$
2x_1+3x_2=5x.
$$

这时把表达式改成：

$$
0x_1+5x_2=5x
$$

并不会改变输出。

这就是权重之间的可补偿关系。OBD 的对角 Hessian 近似基本假设每个权重独立变化，而 OBS 则认为权重之间的耦合关系很重要，应该在剪枝时被保留下来。

## OBD 忽略了什么

假设损失函数关于参数的 Hessian 为：

$$
H=
\begin{bmatrix}
2 & 0.8 & 0.2\\
0.8 & 3 & 0.5\\
0.2 & 0.5 & 4
\end{bmatrix}.
$$

OBD 只保留对角元素：

$$
H_{\mathrm{OBD}}\approx
\begin{bmatrix}
2 & 0 & 0\\
0 & 3 & 0\\
0 & 0 & 4
\end{bmatrix}.
$$

于是它只关心每个权重自身方向上的曲率：

$$
H_{11},\quad H_{22},\quad H_{33}.
$$

但是完整 Hessian 中的非对角元素：

$$
H_{12},\quad H_{13},\quad H_{23}
$$

描述的是不同权重之间的耦合关系。它们决定了当一个权重被删除时，其他权重能否通过联动调整来降低损失增量。

OBS 的出发点就是：如果这些非对角项并不小，就不应该把它们扔掉。

## OBS 的目标函数

设模型参数为：

$$
\mathbf{w}=[w_1,w_2,\ldots,w_n]^T.
$$

如果要删除第 $q$ 个权重，就需要强制：

$$
w_q+\delta w_q=0.
$$

也就是：

$$
\delta w_q=-w_q.
$$

用单位向量 $\mathbf{e}_q$ 表示第 $q$ 个坐标方向，上面的删除约束可以写成：

$$
\mathbf{e}_q^T\delta\mathbf{w}+w_q=0.
$$

在模型已经训练到局部最优点附近时，一阶梯度近似为零，损失变化可以用二阶项近似：

$$
\Delta L \approx
\frac{1}{2}\delta\mathbf{w}^TH\delta\mathbf{w}.
$$

因此 OBS 的优化问题是：

$$
\min_{\delta\mathbf{w}}
\frac{1}{2}\delta\mathbf{w}^TH\delta\mathbf{w},
\qquad
\mathbf{e}_q^T\delta\mathbf{w}+w_q=0.
$$

这个式子非常关键：删除 $w_q$ 是硬约束，但其他参数的变化 $\delta w_i$ 可以自由选择。OBS 要找的是满足删除约束时，让二阶损失增量最小的整体参数扰动。

## 二维例子：怎么算出最优补偿

考虑两个权重：

$$
\mathbf{w}=
\begin{bmatrix}
w_1\\
w_2
\end{bmatrix}.
$$

假设要删除 $w_1$，且：

$$
w_1=1.
$$

那么必须有：

$$
\Delta w_1=-1.
$$

再假设 Hessian 为：

$$
H=
\begin{bmatrix}
2 & 1\\
1 & 2
\end{bmatrix}.
$$

二阶损失增量为：

$$
\Delta L =
\frac{1}{2}
\begin{bmatrix}
\Delta w_1 & \Delta w_2
\end{bmatrix}
\begin{bmatrix}
2 & 1\\
1 & 2
\end{bmatrix}
\begin{bmatrix}
\Delta w_1\\
\Delta w_2
\end{bmatrix}.
$$

展开得到：

$$
\Delta L =
\Delta w_1^2+\Delta w_1\Delta w_2+\Delta w_2^2.
$$

把删除约束 $\Delta w_1=-1$ 代入：

$$
\Delta L =
1-\Delta w_2+\Delta w_2^2.
$$

现在问题变成了一个一元二次函数最小化：

$$
\min_{\Delta w_2}
\left(1-\Delta w_2+\Delta w_2^2\right).
$$

对 $\Delta w_2$ 求导：

$$
\frac{\partial \Delta L}{\partial \Delta w_2} =
-1+2\Delta w_2.
$$

令导数为零：

$$
-1+2\Delta w_2=0.
$$

得到：

$$
\Delta w_2=0.5.
$$

所以 OBS 会选择：

$$
\Delta w_1=-1,\qquad \Delta w_2=0.5.
$$

也就是：

$$
\begin{bmatrix}
w_1\\
w_2
\end{bmatrix}
\longrightarrow
\begin{bmatrix}
0\\
w_2+0.5
\end{bmatrix}.
$$

这就是“删除 $w_1$，同时让 $w_2$ 进行最优补偿”。

## 为什么这个补偿是最优的

继续使用上面的损失函数：

$$
\Delta L=1-\Delta w_2+\Delta w_2^2.
$$

不同补偿量对应的损失增量如下：

| $\Delta w_2$ | $\Delta L$ |
| ---: | ---: |
| $0$ | $1.00$ |
| $0.2$ | $0.84$ |
| $0.5$ | $0.75$ |
| $0.8$ | $0.84$ |
| $1.0$ | $1.00$ |

完全不补偿时，损失增量为 $1$；补偿过头时，损失增量也会变大。最低点正好出现在：

$$
\Delta w_2=0.5.
$$

所以 OBS 里的 Optimal 不是经验上的“稍微调一下”，而是在 Hessian 给出的二阶近似下直接求解：

$$
\arg\min_{\delta\mathbf{w}}\Delta L.
$$

## OBS 的闭式解

对约束优化问题：

$$
\min_{\delta\mathbf{w}}
\frac{1}{2}\delta\mathbf{w}^TH\delta\mathbf{w},
\qquad
\mathbf{e}_q^T\delta\mathbf{w}+w_q=0,
$$

使用拉格朗日乘子可以得到最优参数扰动：

$$
\boxed{
\delta\mathbf{w}^* =
-\frac{w_q}{[H^{-1}]_{qq}}H^{-1}\mathbf{e}_q
}.
$$

对应的最小损失增量为：

$$
\boxed{
\Delta L_q^{\mathrm{OBS}} =
\frac{w_q^2}{2[H^{-1}]_{qq}}
}.
$$

这两个式子分别回答了 OBS 的两个问题：

1. $\delta\mathbf{w}^*$ 告诉我们删除 $w_q$ 后，所有其他权重应该怎么补偿；
2. $\Delta L_q^{\mathrm{OBS}}$ 告诉我们删除 $w_q$ 并完成最优补偿后，预计最少会增加多少损失。

因此，OBS 的剪枝流程可以概括为：

1. 对每个候选权重 $w_q$，计算 $\Delta L_q^{\mathrm{OBS}}$；
2. 选择损失增量最小的权重：

$$
q^*=\arg\min_q\Delta L_q^{\mathrm{OBS}};
$$

3. 使用 $\delta\mathbf{w}^*$ 更新所有权重；
4. 被选中的权重变为零，其余权重完成补偿。

## 与 OBD 的直接对比

OBD 的权重显著性为：

$$
S_q^{\mathrm{OBD}} =
\frac{1}{2}H_{qq}w_q^2.
$$

它只使用 Hessian 对角元素 $H_{qq}$。这意味着 OBD 只估计“单独把 $w_q$ 置零”造成的损失增量，不显式考虑其他权重的补偿。

OBS 的显著性为：

$$
S_q^{\mathrm{OBS}} =
\frac{w_q^2}{2[H^{-1}]_{qq}}.
$$

它使用的是 Hessian 逆矩阵的对角元素 $[H^{-1}]_{qq}$。因为 $H^{-1}$ 已经包含了完整 Hessian 中的参数耦合信息，所以 OBS 的评分天然考虑了“其他参数可以怎样一起调整”。

二者的区别可以总结为：

| 方法 | 使用的二阶信息 | 是否考虑补偿 | 剪枝评分 |
| --- | --- | --- | --- |
| OBD | Hessian 对角元素 $H_{qq}$ | 否 | $\frac{1}{2}H_{qq}w_q^2$ |
| OBS | Hessian 逆的对角元素 $[H^{-1}]_{qq}$ | 是 | $\frac{w_q^2}{2[H^{-1}]_{qq}}$ |

如果 Hessian 近似是对角矩阵，OBS 会退化到接近 OBD 的形式；但当非对角元素较大时，OBS 能利用权重之间的相关性，得到更合理的删除顺序和补偿方向。

## 计算代价与边界

OBS 的代价也比 OBD 高得多。对于 $N$ 个参数，完整 Hessian 的形状是：

$$
N\times N.
$$

直接存储和求逆的代价分别非常高：

$$
O(N^2)\quad \text{存储},
\qquad
O(N^3)\quad \text{直接求逆}.
$$

这也是 OBS 在现代大模型上很难直接原样应用的原因。实际系统通常需要使用块级近似、低秩近似、分层处理、迭代更新或只在局部模块内构造二阶信息。

OBS 的推导还依赖几个前提：

- 模型已经训练到局部最优点附近，一阶梯度可以近似忽略；
- 当前参数附近的损失曲面可以由二阶泰勒展开较好近似；
- Hessian 或其逆矩阵的近似足够稳定；
- 剪枝扰动不能过大，否则局部二阶近似会失效。

因此，OBS 更适合作为理解二阶剪枝和权重补偿的理论基准，而不是可以直接照搬到超大模型上的完整工程方案。

## 总结

OBS 的核心不是“更复杂地计算权重大小”，而是把剪枝看成一个带约束的二阶优化问题：

```text
必须删除某个权重
  -> 其他权重允许共同调整
  -> 用完整 Hessian 描述权重之间的耦合
  -> 求解最小二阶损失增量
  -> 同时得到删除对象和补偿方向
```

OBD 的简化让每个权重可以独立打分，计算更便宜，但会丢掉参数之间的耦合关系。OBS 则更像“外科手术”：切掉一个连接以后，还会调整周围连接，使网络功能尽量保持不变。

最终可以记住这两个公式：

$$
S_q^{\mathrm{OBD}} =
\frac{1}{2}H_{qq}w_q^2,
\qquad
S_q^{\mathrm{OBS}} =
\frac{w_q^2}{2[H^{-1}]_{qq}}.
$$

前者对应“独立删除”，后者对应“删除并最优补偿”。这就是 OBS 相比 OBD 多出来的本质能力。

## 参考资料

- [Optimal Brain Damage（NeurIPS Proceedings）](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html)
- [Second Order Derivatives for Network Pruning: Optimal Brain Surgeon（NeurIPS Proceedings）](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)
