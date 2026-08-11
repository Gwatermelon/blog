---
title: "AWQ 详解：激活感知权重量化如何保护重要通道"
date: 2026-08-11
lastmod: 2026-08-11
draft: false
description: "从输出误差而非权重误差出发，推导 AWQ 如何利用激活幅度识别重要通道，并通过等价缩放、尺度搜索与权重裁剪实现硬件友好的低比特权重量化。"
summary: "AWQ 不用 Hessian 补偿已量化权重，而是根据激活幅度识别重要输入通道，在量化前搜索逐通道缩放因子，让全部权重仍可规整地存为 INT4 或 INT3。"
tags: ["模型推理", "LLM Inference", "模型量化", "AWQ", "PTQ", "INT4", "W4A16"]
categories: ["模型推理"]
math: true
ShowToc: true
TocOpen: true
---

> **AWQ 的核心不是量化后的二阶误差补偿，而是先用激活识别重要权重通道，再通过等价缩放降低这些通道的相对量化误差。**

AWQ，全称 Activation-aware Weight Quantization，即激活感知权重量化，是一种面向大语言模型的训练后量化方法。它不需要反向传播，也不执行 GPTQ 式的逐权重 Hessian 补偿，而是在量化前完成两件事：

1. 根据校准数据中的激活幅度，判断哪些输入通道更重要；
2. 搜索逐输入通道的缩放因子，让重要通道在 INT4 或 INT3 量化下受到更小的相对误差。

AWQ 论文发现，大模型中存在约 $0.1\%$～$1\%$ 的显著权重通道。只保护这一小部分权重，就能显著改善低比特量化后的模型质量；而按照激活幅度选择重要通道，明显优于按照权重绝对值选择。

AWQ 主要面向低比特、仅权重量化。部署中常见的是 W4A16：

```text
Weight      FP16/BF16 -> INT4
Activation  FP16/BF16 -> FP16/BF16
```

激活参与判断权重的重要性，但不代表激活本身必须量化。论文也评估了 INT3，因此 AWQ 并不等同于 INT4；W4A16 只是当前常见的使用方式。

## AWQ 想解决什么问题

先看一个最简单的线性计算：

$$
y=wx.
$$

假设权重为：

$$
w=0.26,
$$

量化步长为：

$$
\Delta=0.1.
$$

使用 Round-to-Nearest，简称 RTN：

$$
Q(w)=\Delta\cdot\operatorname{Round}\left(\frac{w}{\Delta}\right).
$$

于是：

$$
Q(0.26)=0.1\times\operatorname{Round}(2.6)=0.3.
$$

权重量化误差为：

$$
e_w=Q(w)-w=0.3-0.26=0.04.
$$

到这里就是普通 RTN。但 AWQ 注意到，$0.04$ 这个权重误差本身不能说明它对模型的影响有多严重，因为模型真正计算的是 $wx$。

如果：

$$
x=0.01,
$$

输出误差只有：

$$
e_y=e_wx=0.04\times0.01=0.0004.
$$

如果同一个权重对应的激活是：

$$
x=10,
$$

输出误差就变成：

$$
e_y=0.04\times10=0.4.
$$

所以，权重是否重要不能只看 $w$ 本身，还要看它对应的激活是否经常具有较大幅度。这正是 Activation-aware 这个名字的来源。

## 为什么大激活对应的权重更重要

考虑线性层：

$$
\boldsymbol{y}=W\boldsymbol{x}.
$$

第 $i$ 个输出可以展开为：

$$
y_i=w_{i1}x_1+w_{i2}x_2+w_{i3}x_3+\cdots.
$$

假设输入激活为：

$$
\boldsymbol{x}=
\begin{bmatrix}
0.01 & 0.02 & 10 & 0.03
\end{bmatrix}^{T}.
$$

其中 $x_3=10$，远大于其他通道。如果 $w_{i3}$ 产生相同大小的量化误差：

$$
\delta w_{i3}=0.03,
$$

它引起的输出误差为：

$$
\delta y_i=\delta w_{i3}x_3=0.03\times10=0.3.
$$

而在 $x_1=0.01$ 的通道上，同样的权重误差只会产生：

$$
0.03\times0.01=0.0003.
$$

两者相差 $1000$ 倍。因此，经常出现大激活的输入通道，其对应的整列权重更值得保护。这里的“权重通道”不是单独一个权重，而是线性层 $W$ 中与某个输入维度对应的一列。

论文在 INT3、group size 为 128 的 OPT-6.7B 实验中给出了直接证据：

| 方法 | WikiText PPL |
| --- | ---: |
| FP16 | 10.86 |
| 普通 RTN | 23.54 |
| 根据激活保护 $0.1\%$ 权重 | 11.58 |
| 根据激活保护 $1\%$ 权重 | 11.39 |
| 根据权重幅度保护 $1\%$ 权重 | 22.37 |

按照激活选择少量重要权重时，困惑度接近 FP16；按照权重幅度选择则几乎没有解决退化问题。

## 为什么不直接保留少量 FP16 权重

最直接的办法是使用混合精度：

```text
99% 权重 -> INT3/INT4
 1% 权重 -> FP16
```

这种方法在模型质量上很有效，但会产生不规则的数据布局：

```text
INT4 INT4 INT4 FP16 INT4 INT4 FP16 INT4 ...
```

原本规整的低比特矩阵乘法，现在需要在同一权重流中处理两种位宽。它会影响：

- 权重打包；
- 向量化加载；
- GEMM kernel；
- 连续内存访问。

AWQ 因此提出一个更适合硬件的问题：能否让最终所有权重仍然都是 INT4 或 INT3，同时让重要权重不容易受到量化伤害？答案就是逐通道 Scaling。

## AWQ 最关键的技巧：Scaling

仍然从标量计算开始：

$$
y=wx.
$$

AWQ 对它执行一个等价变换。将权重乘以 $s$，同时将激活除以 $s$：

$$
w'=sw,
\qquad
x'=\frac{x}{s}.
$$

于是：

$$
w'x'=(sw)\frac{x}{s}=wx.
$$

没有量化时，模型输出完全不变。

假设：

$$
w=0.26,
\qquad
x=10,
\qquad
s=2.
$$

缩放后：

$$
w'=0.52,
\qquad
x'=5,
$$

仍然有：

$$
w'x'=0.52\times5=2.6.
$$

### 普通量化的误差

当 $\Delta=0.1$ 时：

$$
Q(0.26)=0.3,
$$

量化后的输出为：

$$
Q(w)x=0.3\times10=3.0.
$$

正确输出是 $2.6$，所以误差为 $0.4$。

### 缩放后的误差

如果缩放没有明显改变所在量化组的最大值，可以近似认为：

$$
\Delta'\approx\Delta=0.1.
$$

于是：

$$
Q(0.52)=0.5,
$$

量化后的输出为：

$$
Q(w')x'=0.5\times5=2.5.
$$

现在误差只有 $0.1$：

```text
普通量化：2.6 -> 3.0，误差 0.4
AWQ 缩放：2.6 -> 2.5，误差 0.1
```

这就是 AWQ 最核心的数学技巧。

## 为什么 Scaling 会降低误差

Scaling 能降低误差，不是因为“权重越大越容易量化”，而是因为放大后的权重在计算时还要通过 $x/s$ 抵消。如果量化步长没有同步放大 $s$ 倍，映射回原权重空间后的有效量化间隔会缩小。

### 从等效权重理解

缩放并量化后的计算是：

$$
\widehat{y}=Q(sw)\frac{x}{s}
=\left(\frac{Q(sw)}{s}\right)x.
$$

因此可以把 AWQ 理解为使用了新的等效权重：

$$
\widehat{w}_{\mathrm{AWQ}}=\frac{Q(sw)}{s}.
$$

在前面的例子中：

$$
0.26\xrightarrow{\times2}0.52
\xrightarrow{Q}0.50
\xrightarrow{\div2}0.25.
$$

普通量化只能把 $0.26$ 表示为 $0.30$，误差为 $0.04$；AWQ 的等效权重是 $0.25$，误差为 $0.01$。

原量化网格的间隔是 $0.1$：

$$
\ldots,0.1,0.2,0.3,0.4,\ldots
$$

先乘以 2、量化、再除以 2 后，从原权重空间看，等效网格变成：

$$
\ldots,0.20,0.25,0.30,\ldots
$$

有效间隔从 $0.1$ 变成 $0.05$。一般地：

$$
\Delta_{\mathrm{effective}}=\frac{\Delta'}{s}.
$$

### 从输出误差推导

对量化函数：

$$
Q(w)=\Delta\cdot\operatorname{Round}\left(\frac{w}{\Delta}\right),
$$

可以把舍入误差写成：

$$
Q(w)-w
=\Delta\cdot\operatorname{RoundErr}\left(\frac{w}{\Delta}\right).
$$

普通量化产生的输出误差为：

$$
\operatorname{Err}
=\Delta\cdot\operatorname{RoundErr}\left(\frac{w}{\Delta}\right)x.
$$

缩放后的输出误差为：

$$
\operatorname{Err}'
=\Delta'\cdot\operatorname{RoundErr}\left(\frac{sw}{\Delta'}\right)\frac{x}{s}.
$$

如果两次舍入误差的统计尺度相近，则新旧误差之比近似为：

$$
\frac{\operatorname{Err}'}{\operatorname{Err}}
\approx
\frac{\Delta'}{\Delta}\cdot\frac{1}{s}.
$$

当 $\Delta'\approx\Delta$ 且 $s\gt1$ 时：

$$
\frac{\operatorname{Err}'}{\operatorname{Err}}
\approx\frac{1}{s}\lt1.
$$

因此显著通道的相对误差会下降。这个推导是一种尺度近似，因为缩放前后的具体舍入误差不一定相同。

## 为什么不能把缩放因子设得特别大

前面的收益依赖一个重要前提：$\Delta'$ 没有随着 $s$ 同比例增大。

在 group-wise 对称量化中，一组权重通常共享量化步长。若整数范围采用 $[-q_{\max},q_{\max}]$，则：

$$
\Delta=\frac{\max\lvert w\rvert}{q_{\max}}.
$$

假设一组权重原来是：

```text
0.1  0.3  0.5  0.7
```

如果把其中某个权重过度放大：

```text
0.1  0.3  5.0  0.7
```

新的组内最大值会把 $\Delta'$ 撑大，导致同组其他非显著权重使用更粗的量化网格。

所以 AWQ 面临一个平衡：

- $s$ 太小，重要通道保护不够；
- $s$ 太大，组内量化步长变大，其他权重误差上升。

论文在 OPT-6.7B 的 INT3-g128 实验中也观察到这一点：固定 $s=2$ 时 PPL 从 RTN 的 23.54 降至 11.92；继续增大到 $s=4$，PPL 反而回升到 12.36。因此，AWQ 需要搜索合适的逐通道缩放因子，而不是无限放大重要通道。

## AWQ 如何找到缩放因子

AWQ 使用一小批校准数据运行 FP16 模型，收集每个线性层的输入激活 $X$。对于输入通道 $j$，统计平均激活幅度：

$$
a_j=\operatorname{mean}\left(\lvert X_j\rvert\right).
$$

如果某些通道的 $a_j$ 明显更大，就说明这些通道更可能携带重要特征。AWQ 用激活统计构造逐通道缩放因子的搜索空间：

$$
s_j=a_j^{\alpha},
\qquad
0\le\alpha\le1.
$$

$\alpha=0$ 表示不缩放；$\alpha=1$ 表示采用搜索空间中最激进的激活感知缩放。算法在 $[0,1]$ 上对 $\alpha$ 做快速网格搜索。

令：

$$
S=\operatorname{diag}(\boldsymbol{s}),
$$

则权重和激活的等价变换是：

$$
W'=WS,
\qquad
X'=S^{-1}X.
$$

没有量化时：

$$
W'X'=WSS^{-1}X=WX.
$$

搜索目标是让量化后的层输出尽量接近原始 FP16 层输出：

$$
\boldsymbol{s}^{*}
=\underset{\boldsymbol{s}}{\operatorname{argmin}}
\left\lVert
Q\!\left(W\operatorname{diag}(\boldsymbol{s})\right)
\operatorname{diag}(\boldsymbol{s})^{-1}X
-WX
\right\rVert_{2}.
$$

结合 $s_j=a_j^{\alpha}$ 的搜索空间，可以写成：

$$
\alpha^{*}
=\underset{0\le\alpha\le1}{\operatorname{argmin}}
\mathcal{L}\!\left(\boldsymbol{a}^{\alpha}\right).
$$

这一步不需要反向传播。AWQ 只对少量候选 $\alpha$ 模拟量化，比较量化输出与 FP16 输出的差异，选择误差最小的候选值。论文还在缩放搜索之后加入权重裁剪搜索，以进一步降低量化 MSE。

## 完整 AWQ 流程

假设现在要量化一个线性层权重 $W$，完整流程可以整理为六步。

### 第一步：运行校准数据

用一小批校准样本运行 FP16 模型，缓存当前线性层的输入激活：

$$
X.
$$

### 第二步：统计输入通道的激活幅度

对每个输入通道计算：

$$
a_j=\operatorname{mean}\left(\lvert X_j\rvert\right).
$$

这些统计量用于判断哪些输入通道经常产生大激活。

### 第三步：构造 Scaling

根据候选 $\alpha$ 计算：

$$
s_j=a_j^{\alpha}.
$$

### 第四步：搜索最优 $\alpha$

对各个候选值模拟量化，使：

$$
\left\lVert Y_{\mathrm{quant}}-Y_{\mathrm{FP16}}\right\rVert_{2}
$$

最小。

### 第五步：执行等价变换

对权重应用逐输入通道缩放：

$$
W'=W\operatorname{diag}(\boldsymbol{s}).
$$

理论上，对输入应用逆缩放：

$$
X'=\operatorname{diag}(\boldsymbol{s})^{-1}X.
$$

工程实现中，逆缩放通常可以融合进前一个算子或归一化层，因此不一定需要在推理时单独增加一次逐元素除法。

### 第六步：裁剪并量化缩放后的权重

对 $W'$ 搜索合适的 clipping 范围，然后执行 group-wise INT4 或 INT3 量化。最终模型保持统一的低比特权重格式，例如 W4A16，而不是不规则的“99% INT4 + 1% FP16”。

## AWQ 与 GPTQ 的区别

AWQ 和 GPTQ 都属于 PTQ，也都常用于 W4A16，但两者解决量化误差的方式不同：

| 对比项 | GPTQ | AWQ |
| --- | --- | --- |
| 类型 | PTQ | PTQ |
| 常见形式 | W4A16 | W4A16 |
| 是否需要校准数据 | 需要 | 需要 |
| 核心思想 | 量化后补偿后续权重误差 | 量化前保护重要输入通道 |
| 重要性来源 | 二阶或 Hessian 近似信息 | 激活通道平均幅度 |
| 是否逐序量化 | 是 | 不是核心步骤 |
| GPTQ 式误差补偿 | 有 | 无 |
| Scaling | 不是核心机制 | 核心机制 |
| 是否需要反向传播 | 不需要 | 不需要 |

可以把差异概括为：GPTQ 关注“量化一个权重后，怎样调整其他权重以补偿误差”；AWQ 关注“量化之前，怎样根据激活识别并保护更重要的权重通道”。

## 边界与结论

AWQ 的价值来自三个条件同时成立：

1. 模型中只有少量输入通道对量化误差特别敏感；
2. 激活幅度能够为通道重要性提供有效信号；
3. 缩放后组内量化步长不会增大到抵消保护收益。

它并不保证 INT4 一定带来固定倍数的推理加速。真实收益还取决于权重打包格式、反量化 kernel、矩阵形状、显存带宽和部署框架。AWQ 解决的是如何在硬件友好的统一低比特格式下保留模型质量；要把压缩转化为吞吐和延迟收益，仍然需要对应的推理内核。

最终可以记住三点：

- **重要性来自激活。** 同样大小的权重误差，乘上大激活后会造成更大的输出误差。
- **保护手段是等价缩放。** 放大重要权重通道，同时对激活做逆缩放，使无量化时的函数保持不变。
- **缩放因子必须搜索。** 过小保护不足，过大会撑大量化组的步长，因此 AWQ 用层输出误差选择平衡点。

## 参考资料

1. [Lin et al., AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html)
2. [AWQ 官方实现：mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)
3. [Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
4. [模型量化基础：从浮点表示到整数矩阵乘法](/model-inference/model-quantization-fundamentals/)
5. [大模型量化粒度详解：逐层、逐通道与逐组量化](/model-inference/quantization-granularity/)
