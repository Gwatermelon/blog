---
title: "self-attention"
date: 2026-07-23
lastmod: 2026-07-23
draft: false
description: "从 Q、K、V 和缩放点积出发理解 Self-Attention，进一步分析因果掩码、增量解码、KV Cache，以及为什么没有 Q Cache。"
summary: "系统梳理 Self-Attention 的输入、QKV 投影、注意力权重与因果掩码，并从矩阵变化解释增量解码为何只计算最后一行、为何缓存 K 和 V。"
tags: ["AI 基础", "Transformer", "Self-Attention", "KV Cache", "因果掩码"]
categories: ["AI 基础原理"]
math: true
ShowToc: true
TocOpen: true
---

Self-Attention 要解决的核心问题是：让序列中的每个 token 根据当前上下文，动态地从其他 token 中提取自己需要的信息。

例如句子：

> 小明把书放在桌子上，因为它很重。

模型处理“它”时，需要判断“它”更可能指“书”，而不是“小明”或“桌子”。Self-Attention 会让“它”与序列中的其他 token 计算相关性，再根据相关性对这些 token 携带的信息加权汇总。

之所以叫 Self-Attention，是因为生成 Query、Key、Value 的输入都来自同一个序列。Cross-Attention 则不同：Query 与 Key、Value 来自不同序列。

## Self-Attention 的输入

假设输入序列有 $n$ 个 token，每个 token 的隐藏向量维度为 $d_{\text{model}}$。将这些向量按行排列，得到输入矩阵：

$$
X\in\mathbb{R}^{n\times d_{\text{model}}}.
$$

例如有 4 个 token，每个 token 是一个 8 维向量，那么：

$$
X\in\mathbb{R}^{4\times 8}.
$$

这里的 $X$ 不一定只是最初的 Token Embedding：

- 第一层 Transformer 的输入通常是 Token Embedding 加位置信息；
- 后续层的输入是上一层输出的隐藏状态。

因此，每一层 Self-Attention 都会重新理解当前序列表示。

## Q、K、V 是怎么得到的

输入矩阵 $X$ 分别乘以三个可学习的权重矩阵：

$$
\begin{aligned}
Q&=XW_Q,\\
K&=XW_K,\\
V&=XW_V.
\end{aligned}
$$

其中：

$$
W_Q\in\mathbb{R}^{d_{\text{model}}\times d_k},
\qquad
W_K\in\mathbb{R}^{d_{\text{model}}\times d_k},
\qquad
W_V\in\mathbb{R}^{d_{\text{model}}\times d_v}.
$$

因此：

$$
Q\in\mathbb{R}^{n\times d_k},
\qquad
K\in\mathbb{R}^{n\times d_k},
\qquad
V\in\mathbb{R}^{n\times d_v}.
$$

每个 token 的输入向量 $x_i$ 会被投影成三种不同的表示：

$$
x_i\longrightarrow q_i,\ k_i,\ v_i.
$$

这三种向量不是同一份信息的简单复制，而是模型通过训练学习出的三种不同表示。

## Q、K、V 分别有什么作用

可以把 Self-Attention 类比成一个检索系统。

### Query：当前 token 想找什么

Query 表示当前 token 想从上下文中寻找什么信息。例如当前 token 是“它”，它的 Query 可能倾向于寻找：

- 前面出现过的名词；
- 与“重”这个属性匹配的对象；
- 语法上可能作为指代对象的词。

这只是功能上的直觉。模型并没有显式规定某个维度必须表示“名词”或“重量”，这些模式都是从训练数据中学习得到的。

### Key：当前 token 可以如何被匹配

Key 表示当前 token 可以通过哪些特征被其他 token 匹配到。例如“书”的 Key 可能编码：

- 它是名词；
- 它可以被代词指代；
- 它可能具有重量；
- 它在当前句子中的语法角色。

Query 和 Key 的匹配程度决定 Attention 分数。

### Value：当前 token 实际提供什么

Value 表示：如果其他 token 关注当前 token，它应该向对方传递什么内容。

可以把三者简化为：

| 向量 | 作用 |
| --- | --- |
| Query | 当前 token 要找什么 |
| Key | 当前 token 可以如何被匹配 |
| Value | 当前 token 实际提供什么内容 |

Key 决定“是否应该找到我”，Value 决定“找到我之后拿走什么信息”。同样，这些只是功能解释，并不表示模型内部存在人工定义的查询字段或内容字段。

## 单个 token 的 Attention 计算

假设正在计算第 $i$ 个 token 的输出，它的 Query 为：

$$
q_i\in\mathbb{R}^{d_k}.
$$

序列中第 $j$ 个 token 的 Key 为：

$$
k_j\in\mathbb{R}^{d_k}.
$$

先计算二者的点积：

$$
s_{ij}=q_i^{\mathsf T}k_j.
$$

$s_{ij}$ 表示第 $i$ 个 token 对第 $j$ 个 token 的关注程度。点积越大，通常说明 $q_i$ 和 $k_j$ 越匹配。

对第 $i$ 个 token，需要分别与序列中的所有 Key 计算：

$$
s_{i1},s_{i2},\ldots,s_{in}.
$$

经过 Softmax 后得到 Attention 权重：

$$
a_{ij}
=\frac{\exp(s_{ij})}
{\sum_{m=1}^{n}\exp(s_{im})}.
$$

这些权重满足：

$$
\sum_{j=1}^{n}a_{ij}=1.
$$

最后使用这些权重对 Value 加权求和：

$$
o_i=\sum_{j=1}^{n}a_{ij}v_j.
$$

因此，Attention 的核心过程可以概括为：

1. 用 Query 和所有 Key 计算相关性；
2. 用 Softmax 把相关性转换为权重；
3. 用这些权重加权汇总 Value。

## 为什么要除以 $\sqrt{d_k}$

实际使用的 Attention 分数不是简单的 $q_i^{\mathsf T}k_j$，而是：

$$
s_{ij}=\frac{q_i^{\mathsf T}k_j}{\sqrt{d_k}}.
$$

点积可以展开为：

$$
q_i^{\mathsf T}k_j
=\sum_{r=1}^{d_k}q_{ir}k_{jr}.
$$

如果 Query 和 Key 的各个分量相互独立、均值接近 0、方差接近 1，那么点积的方差大约会随 $d_k$ 线性增长。$d_k$ 越大，分数的绝对值通常越大。

如果输入 Softmax 的分数差异过大，例如：

$$
[1,2,20],
$$

Softmax 输出会非常接近：

$$
[0,0,1].
$$

这会使 Softmax 进入饱和区域，梯度变小，不利于稳定训练。除以 $\sqrt{d_k}$ 后，点积分数的方差被拉回相对稳定的尺度，从而减轻 Softmax 过度饱和的问题。

## Self-Attention 的矩阵形式

把所有 token 的计算合并起来，缩放点积分数矩阵为：

$$
S=\frac{QK^{\mathsf T}}{\sqrt{d_k}},
\qquad
S\in\mathbb{R}^{n\times n}.
$$

第 $i$ 行记录第 $i$ 个 token 对所有 token 的注意力分数。对每一行执行 Softmax：

$$
A=\operatorname{Softmax}(S),
\qquad
A\in\mathbb{R}^{n\times n}.
$$

再与 Value 矩阵相乘：

$$
O=AV,
\qquad
O\in\mathbb{R}^{n\times d_v}.
$$

合起来就是经典的缩放点积注意力：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{Softmax}\!\left(
\frac{QK^{\mathsf T}}{\sqrt{d_k}}
\right)V.
$$

在多头注意力中，$Q$、$K$、$V$ 会被拆分到多个头中。每个头独立执行缩放点积注意力，各头输出拼接后再经过一次线性投影。不同注意力头可以学习不同类型的匹配关系。

## 掩码矩阵是什么

Decoder-only 大语言模型采用自回归生成：

$$
x_1\longrightarrow x_2\longrightarrow x_3\longrightarrow\cdots.
$$

预测第 $i$ 个位置时，模型不能看到未来 token。否则训练时模型在预测目标时可以直接读取后面的答案，造成信息泄漏。

因此需要使用 Causal Mask，也叫因果掩码。对于长度为 $n$ 的序列，掩码矩阵定义为：

$$
M_{ij}
=\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

以长度 4 为例：

$$
M=
\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}.
$$

在 Softmax 之前把掩码加到分数矩阵：

$$
A
=\operatorname{Softmax}\!\left(
\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M
\right).
$$

因为 $\exp(-\infty)=0$，被屏蔽的位置经过 Softmax 后权重为 0。最终的 Attention 权重矩阵呈下三角形式：

$$
A=
\begin{bmatrix}
1&0&0&0\\
a_{21}&a_{22}&0&0\\
a_{31}&a_{32}&a_{33}&0\\
a_{41}&a_{42}&a_{43}&a_{44}
\end{bmatrix}.
$$

它表示：

- token 1 只能看到 token 1；
- token 2 可以看到 token 1、2；
- token 3 可以看到 token 1、2、3；
- token 4 可以看到 token 1、2、3、4。

在语言模型训练中，位置 $i$ 的隐藏状态通常用来预测位置 $i+1$ 的 token。例如输入“我 喜欢 吃”，模型在“吃”这个位置得到的隐藏状态可以用来预测下一个 token“苹果”。

## 完整的 QKV 计算过程

假设某层 Transformer 当前有 $n$ 个 token，其输入隐藏状态为：

$$
X_n\in\mathbb{R}^{n\times d_{\text{model}}}.
$$

经过线性投影：

$$
\begin{aligned}
Q_n&=X_nW_Q,\\
K_n&=X_nW_K,\\
V_n&=X_nW_V.
\end{aligned}
$$

然后计算缩放点积分数：

$$
S_n=\frac{Q_nK_n^{\mathsf T}}{\sqrt{d_k}}.
$$

加入因果掩码并执行 Softmax：

$$
A_n=\operatorname{Softmax}(S_n+M_n).
$$

最后得到：

$$
O_n=A_nV_n.
$$

$A_n\in\mathbb{R}^{n\times n}$，其中第 $i$ 行表示第 $i$ 个 token 对它可以看到的各个 token 分配了多少注意力。

## 生成一个新 token 后，矩阵发生什么变化

假设当前序列为：

$$
x_1,x_2,\ldots,x_n,
$$

模型又生成了一个 token $x_{n+1}$。从完整矩阵的概念上看，下一轮输入变成：

$$
X_{n+1}=
\begin{bmatrix}
X_n\\
x_{n+1}
\end{bmatrix}.
$$

模型参数 $W_Q$、$W_K$、$W_V$ 在推理期间不会发生变化，于是：

$$
Q_{n+1}
=X_{n+1}W_Q
=\begin{bmatrix}
Q_n\\
q_{n+1}
\end{bmatrix},
$$

$$
K_{n+1}=
\begin{bmatrix}
K_n\\
k_{n+1}
\end{bmatrix},
\qquad
V_{n+1}=
\begin{bmatrix}
V_n\\
v_{n+1}
\end{bmatrix}.
$$

新的 Q、K、V 相比旧矩阵都只多出最后一行。但这不意味着三者都值得缓存。

## 从完整 Attention 矩阵观察新增行和新增列

如果概念上展开新的未缩放分数矩阵：

$$
Q_{n+1}K_{n+1}^{\mathsf T}
=\begin{bmatrix}
Q_nK_n^{\mathsf T} & Q_nk_{n+1}^{\mathsf T}\\
q_{n+1}K_n^{\mathsf T} & q_{n+1}k_{n+1}^{\mathsf T}
\end{bmatrix}.
$$

这个矩阵可以分成四块。

### 左上角：历史 Attention 分数

$$
Q_nK_n^{\mathsf T}\in\mathbb{R}^{n\times n}.
$$

这是上一轮已经计算过的历史 token 之间的分数。历史位置的最终隐藏状态已经产生，不需要重新生成，因此无需再计算这一整块。

### 右上角：历史 Query 与新增 Key

$$
Q_nk_{n+1}^{\mathsf T}\in\mathbb{R}^{n\times 1}.
$$

这一列表示历史位置是否关注新 token。但对每个历史位置来说，第 $n+1$ 个 token 都属于未来位置，会被因果掩码屏蔽：

$$
Q_nk_{n+1}^{\mathsf T}
\longrightarrow
\begin{bmatrix}
-\infty\\
\vdots\\
-\infty
\end{bmatrix}.
$$

Softmax 后这一列全部变成 0，因此没有必要使用历史 Query 计算这列。

### 左下角：新增 Query 关注历史 Key

$$
q_{n+1}K_n^{\mathsf T}\in\mathbb{R}^{1\times n}.
$$

它表示新 token 对所有历史 token 的注意力分数。这一部分必须计算，因为新 token 需要读取整个历史上下文。计算它需要当前新增的 Query $q_{n+1}$ 和所有历史 Key $K_n$，所以历史 Key 必须保存。

### 右下角：新 token 关注自己

$$
q_{n+1}k_{n+1}^{\mathsf T}.
$$

这是一个标量，表示新 token 对自身位置的注意力分数。当前 token 可以看到自己，因此这一项不会被因果掩码屏蔽。

## 增量解码实际只需要计算最后一行

在真实的增量解码中，不会构造完整的 $(n+1)\times(n+1)$ 注意力矩阵。只需要计算新 token 对历史和自身的最后一行：

$$
s_{n+1}
=\frac{q_{n+1}}{\sqrt{d_k}}
\begin{bmatrix}
K_n\\
k_{n+1}
\end{bmatrix}^{\mathsf T}
\in\mathbb{R}^{1\times(n+1)}.
$$

然后：

$$
a_{n+1}=\operatorname{Softmax}(s_{n+1}),
$$

$$
o_{n+1}
=a_{n+1}
\begin{bmatrix}
V_n\\
v_{n+1}
\end{bmatrix}
\in\mathbb{R}^{1\times d_v}.
$$

因此，解码阶段不是重新构造完整的注意力矩阵，而是直接计算尺寸为 $1\times(n+1)$ 的最后一行。

用 4 个 token 的例子理解：假设前三个 token 已经处理完毕，当前新增第 4 个 token。完整分数矩阵在因果掩码下概念上为：

$$
\begin{bmatrix}
q_1^{\mathsf T}k_1&-\infty&-\infty&-\infty\\
q_2^{\mathsf T}k_1&q_2^{\mathsf T}k_2&-\infty&-\infty\\
q_3^{\mathsf T}k_1&q_3^{\mathsf T}k_2&q_3^{\mathsf T}k_3&-\infty\\
q_4^{\mathsf T}k_1&q_4^{\mathsf T}k_2&q_4^{\mathsf T}k_3&q_4^{\mathsf T}k_4
\end{bmatrix}.
$$

前三行在之前已经计算完成，第四列的前三个元素又会被 Mask 屏蔽。所以这一轮真正需要计算的只有：

$$
\begin{bmatrix}
q_4^{\mathsf T}k_1&
q_4^{\mathsf T}k_2&
q_4^{\mathsf T}k_3&
q_4^{\mathsf T}k_4
\end{bmatrix}.
$$

计算这一行需要新的 $q_4$、历史 $k_1,k_2,k_3$ 和新的 $k_4$，不需要历史 $q_1,q_2,q_3$。

## 为什么缓存 K 和 V，而不缓存 Q

从功能角度看：

- Query 是当前 token 主动提出的问题；
- Key 是历史 token 留给未来查询的索引；
- Value 是历史 token 留给未来读取的内容。

未来 token 到来时，会产生自己的 Query：

$$
q_{\text{future}},
$$

然后用它查询历史 Key，并从历史 Value 中提取信息。过去的 Query 只服务于过去那个位置的 Attention 计算，计算完成后就没有复用价值。

可以类比数据库：

- Query：本次发出的查询语句；
- Key：数据库索引；
- Value：数据库记录。

执行下一次查询时，数据库索引和记录仍然有用，而上一次查询语句通常没有复用价值；下一次会发出新的查询语句。

因此，增量解码缓存的是历史 $K$ 和 $V$，而不是 $Q$。这就是 KV Cache 的来源。

## 总结

Self-Attention 可以概括为以下过程：

1. 输入隐藏状态分别投影为 Query、Key 和 Value；
2. Query 与 Key 的缩放点积给出 token 之间的匹配分数；
3. Softmax 将分数转换为权重，再对 Value 加权汇总；
4. Decoder-only 模型通过因果掩码阻止当前位置读取未来 token；
5. 增量解码只计算新 token 对历史和自身的最后一行注意力；
6. 历史 Key 和 Value 会被未来 token 反复使用，而历史 Query 不会，因此推理时使用 KV Cache，而没有 Q Cache。
