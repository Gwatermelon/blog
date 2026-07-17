---
title: "RoPE 旋转位置编码：用绝对旋转得到相对位置"
date: 2026-07-17
lastmod: 2026-07-17
draft: false
description: "从 Transformer 为什么需要位置信息出发，推导 RoPE 如何通过二维旋转把绝对位置写入 Query 和 Key，并让注意力内积只依赖相对距离。"
summary: "RoPE 不把位置向量加到输入上，而是按位置旋转 Query 和 Key。本文从二维旋转矩阵推导相对位置性质，再解释高维分组、多频率设计、实现方法与常见误区。"
tags: ["AI 基础", "Transformer", "位置编码", "RoPE", "Attention"]
categories: ["AI 基础原理"]
math: true
ShowToc: true
TocOpen: true
---

> **核心直觉：** RoPE 分别按照绝对位置旋转 Query 和 Key，但两者做内积时，绝对角度会相减，最终只留下相对位置。

Transformer 的 Attention 可以建立任意两个 token 之间的联系，却不会天然区分它们的先后顺序。RoPE（Rotary Position Embedding，旋转位置编码）通过旋转 Query 和 Key，把位置直接写进注意力打分过程。它的巧妙之处在于：单独看每个向量时编码的是绝对位置，比较两个向量时得到的却是相对位置。

## 为什么 Transformer 需要位置编码

RNN 按时间步依次处理输入，顺序已经写在计算路径中；卷积也通过局部窗口保留相对邻接关系。标准 Self-Attention 则不同：如果忽略位置编码，同时对输入 token 做同一种排列，输出只会跟着做相同排列。换句话说，Attention 本身无法仅凭内容区分“猫追老鼠”和“老鼠追猫”。

设位置 $m$ 的 token 产生 Query $q_m$，位置 $n$ 的 token 产生 Key $k_n$。注意力分数的核心是：

$$
q_m^\top k_n.
$$

普通线性投影主要携带语义信息，并不知道 $m$、$n$ 分别是多少，也不知道两个 token 相距多远。位置编码要做的，就是让这个分数同时感知内容与顺序。

## 从绝对位置编码说起

绝对位置编码通常把只依赖位置 $k$ 的向量 $p_k$ 加到输入表示 $x_k$ 上：

$$
\tilde{x}_k=x_k+p_k.
$$

常见方案可以分成三类。

### 可训练位置向量

如果模型最大长度为 $L$、隐藏维度为 $d$，可以直接初始化一个 $L\times d$ 的参数矩阵并随模型一起训练。它简单、灵活，但训练时没有见过的位置不存在已学习的向量，因此直接扩展到更长序列并不自然。

### Sinusoidal 位置编码

原始 Transformer 使用固定的正弦—余弦函数：

$$
p_{k,2i}=\sin\left(\frac{k}{10000^{2i/d}}\right),
\qquad
p_{k,2i+1}=\cos\left(\frac{k}{10000^{2i/d}}\right).
$$

其中 $k=0,1,\ldots,L-1$ 是位置编号，$i=0,1,\ldots,d/2-1$ 是二维分量组的编号。不同 $i$ 对应不同频率：高频分量变化快，适合分辨较近位置；低频分量变化慢，能覆盖更长尺度。

它具有显式生成规则，可以计算训练长度以外的位置。三角恒等式还允许位置 $k+r$ 的编码由位置 $k$ 的正弦、余弦分量线性表示，因而包含表达相对位移的可能。不过它仍然是先生成位置向量，再把位置向量加到 token 表示上。

### 递归或连续动力系统位置编码

还可以从初始状态 $p_0$ 出发，通过 $p_{k+1}=f(p_k)$ 递归地产生位置编码。FLOATER 进一步使用神经常微分方程描述位置状态随连续位置的演化，兼顾可学习性与长度外推。但递归生成也可能牺牲并行性。

## RoPE 的设计目标

相对位置编码不必完整描述“这是第几个 token”，而更关心“两个 token 相隔多远、方向如何”。RoPE 的目标可以写成：

$$
\tilde{q}_m=f(q,m),
\qquad
\tilde{k}_n=f(k,n),
$$

并希望变换后的内积满足：

$$
\left\langle f(q,m),f(k,n)\right\rangle
=g(q,k,n-m).
$$

左侧分别把绝对位置 $m$、$n$ 写入两个向量，右侧却只依赖相对位移 $n-m$。这看似矛盾，却可以用钟表来理解：两根指针分别位于绝对角度 $m\theta$ 和 $n\theta$，它们的夹角只等于 $(n-m)\theta$。

## 二维旋转如何得到相对位置

先看二维向量。定义角度为 $m\theta$ 的旋转矩阵：

$$
R_m=
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta)\\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}.
$$

对 Query 和 Key 分别应用各自位置对应的旋转：

$$
\tilde{q}_m=R_mq,
\qquad
\tilde{k}_n=R_nk.
$$

现在计算内积：

$$
\begin{aligned}
\tilde{q}_m^\top\tilde{k}_n
&=(R_mq)^\top(R_nk)\\
&=q^\top R_m^\top R_nk.
\end{aligned}
$$

旋转矩阵是正交矩阵，所以 $R_m^\top=R_{-m}$；连续旋转的角度可以相加，因此：

$$
R_m^\top R_n=R_{-m}R_n=R_{n-m}.
$$

代回内积得到：

$$
\boxed{
\tilde{q}_m^\top\tilde{k}_n
=q^\top R_{n-m}k
}
$$

最终结果不再分别依赖 $m$ 和 $n$，只依赖相对位移 $n-m$。如果交换 Query 与 Key 的定义或采用另一套符号约定，也可能写成 $m-n$；关键不是符号方向，而是只保留两者之差。

旋转还有一个重要性质：

$$
\|R_mq\|_2=\|q\|_2.
$$

RoPE 改变向量方向，却不改变模长。它把位置信息注入 Query 和 Key 的相对夹角，而不是通过缩放任意改变向量大小。

## 高维向量如何旋转

实际注意力头的维度远大于 2。RoPE 不增加维度，而是把偶数维向量按相邻分量两两分组：

$$
(q_0,q_1),(q_2,q_3),\ldots,(q_{d-2},q_{d-1}).
$$

第 $i$ 组使用频率：

$$
\theta_i=10000^{-2i/d},
\qquad i=0,1,\ldots,d/2-1.
$$

位置为 $m$ 时，第 $i$ 组旋转 $m\theta_i$：

$$
\begin{bmatrix}
\tilde{q}_{m,2i}\\
\tilde{q}_{m,2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i)\\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
q_{2i}\\
q_{2i+1}
\end{bmatrix}.
$$

例如 $d=8$ 时共有四组：

| 组编号 $i$ | 维度对 | 频率 $\theta_i$ |
| --- | --- | --- |
| 0 | $(q_0,q_1)$ | $1$ |
| 1 | $(q_2,q_3)$ | $0.1$ |
| 2 | $(q_4,q_5)$ | $0.01$ |
| 3 | $(q_6,q_7)$ | $0.001$ |

可以把它想成四块转速不同的表盘。位置 $m=3$ 时，四组分别旋转 $3$、$0.3$、$0.03$、$0.003$ 弧度。向量仍然只有 8 个分量，但同一个位置同时在多个频率尺度上留下了特征。

如果所有维度对都使用相同频率，它们的位置变化模式会重复，表达尺度也十分单一。多频率设计让近距离变化与长距离变化可以由不同维度共同描述。这与 Sinusoidal 编码的多频率思想相通，但两者的应用位置不同：Sinusoidal 生成向量后与输入相加，RoPE 则直接旋转 Query 和 Key。

## 不构造大矩阵的实现方式

概念上可以把所有二维旋转矩阵放进一个分块对角矩阵，但实际实现无需构造稀疏的 $d\times d$ 矩阵。对每个二维分量对直接计算即可：

$$
\operatorname{RoPE}(q,m)
=q\odot\cos(m\Theta)+\operatorname{rotate\_half}(q)\odot\sin(m\Theta),
$$

其中 $\Theta$ 把每个 $\theta_i$ 重复到对应的两个分量，$\odot$ 表示逐元素乘法。若采用相邻分量配对，则：

$$
\operatorname{rotate\_half}([q_0,q_1,q_2,q_3,\ldots])
=[-q_1,q_0,-q_3,q_2,\ldots].
$$

不同代码库还可能采用“前半维与后半维配对”的布局。只要角度排列和 `rotate_half` 的实现彼此一致，数学性质相同；混用两种布局才会产生错误。

## Python 验证相对位置性质

下面只用 Python 标准库实现相邻分量配对的 RoPE，并验证：同时平移 Query 和 Key 的绝对位置后，只要相对距离不变，旋转后的内积就不变。

```python
from math import cos, isclose, sin


def rope(vector: list[float], position: int, base: float = 10_000.0) -> list[float]:
    """对偶数维向量应用相邻分量配对的 RoPE。"""
    dimension = len(vector)
    if dimension % 2 != 0:
        raise ValueError("RoPE requires an even dimension")

    rotated: list[float] = []
    for pair_index in range(dimension // 2):
        theta = base ** (-2 * pair_index / dimension)
        angle = position * theta
        x = vector[2 * pair_index]
        y = vector[2 * pair_index + 1]
        rotated.extend([
            x * cos(angle) - y * sin(angle),
            x * sin(angle) + y * cos(angle),
        ])
    return rotated


def dot(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right, strict=True))


query = [0.3, -0.7, 1.2, 0.4, -0.2, 0.9, 0.5, -1.1]
key = [-0.6, 0.8, 0.1, 1.3, 0.7, -0.4, 1.0, 0.2]

# 两组绝对位置不同，但相对位移都等于 3。
score_a = dot(rope(query, 10), rope(key, 7))
score_b = dot(rope(query, 100), rope(key, 97))

print(score_a)
print(score_b)
assert isclose(score_a, score_b, rel_tol=1e-12, abs_tol=1e-12)
```

这个验证说明 RoPE 不是让旋转后的向量本身只依赖相对位置。`rope(query, 10)` 与 `rope(query, 100)` 显然不同；只有在 Query 和 Key 进入内积后，绝对位置才合并为相对位移。

## RoPE 放在 Attention 的哪个位置

标准做法是先由输入计算 Query、Key、Value，再对 Query 和 Key 应用 RoPE：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

$$
\tilde{Q}=\operatorname{RoPE}(Q),
\qquad
\tilde{K}=\operatorname{RoPE}(K),
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(
\frac{\tilde{Q}\tilde{K}^\top}{\sqrt{d_h}}
\right)V.
$$

Value 通常不旋转，因为位置依赖已经进入注意力权重；最终仍由这些权重聚合 Value。工程实现中，RoPE 一般按注意力头的 head dimension 计算，并缓存各位置的正弦、余弦值，避免每层重复生成。

## 长度外推应如何理解

RoPE 的角度由显式函数生成，因此可以为训练长度之外的位置计算旋转角。这比固定大小的可训练位置表更自然，但“可以计算”不等于“模型在任意长度上都能可靠工作”。

原因包括：

- 三角函数具有周期性，距离增加后部分频率会出现相位混叠；
- 模型训练时只适应了有限范围内的相对距离分布；
- 高频与低频维度在超长上下文中的分工可能偏离训练阶段；
- 注意力模式、数据分布和数值精度同样影响长上下文表现。

因此，RoPE 提供了长度外推的结构基础，但真正扩展上下文窗口时，通常还要结合频率缩放、位置插值或额外训练，并通过目标长度上的实验验证。

## 常见误区

### RoPE 是把一个位置向量加到输入上

RoPE 的核心不是加法，而是对 Query 和 Key 的二维分量对做乘性旋转。它通常不直接修改输入 embedding，也不改变向量维度。

### 每两个维度会变成一个新向量

二维分组只是为了执行旋转。一个 $d$ 维 Query 经过 RoPE 后仍是同一个 $d$ 维 Query，不会被拆成 $d/2$ 个独立 token 表示。

### 两个维度分别使用两个频率

同一分量对共享同一个 $\theta_i$，因为它们共同构成一个二维平面。频率是在不同二维平面之间变化的。

### RoPE 只依赖相对位置，所以没有绝对位置

单独的 $R_mq$ 明确依赖绝对位置 $m$。只有与 $R_nk$ 做内积时，旋转矩阵才组合成 $R_{n-m}$。准确说法是：RoPE 用绝对旋转参数化 Query 和 Key，并在点积注意力中产生相对位置依赖。

### 有显式公式就能无限外推

公式可以生成任意位置的角度，但模型是否理解训练范围之外的相位组合是另一个问题。数学可定义性不能替代长上下文评测。

## 总结

RoPE 的推导可以浓缩为三步：

1. Self-Attention 需要位置才能区分排列不同、内容相似的序列；
2. 给位置 $m$、$n$ 的 Query 和 Key 分别施加旋转 $R_m$、$R_n$；
3. 利用 $R_m^\top R_n=R_{n-m}$，让注意力内积只依赖相对位移。

高维情况下，RoPE 把向量拆成多个二维分量对，并使用从高到低的不同频率。它不增加维度、不改变向量模长，也不需要显式构造大旋转矩阵。正是这种“绝对位置进入、相对位置生效”的结构，让 RoPE 成为现代大语言模型中非常常见的位置编码方案。

## 参考资料

- [Su et al.：RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [苏剑林：Transformer升级之路——博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265)
- [Vaswani et al.：Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Liu et al.：Learning to Encode Position for Transformer with Continuous Dynamical Model](https://arxiv.org/abs/2003.09229)
