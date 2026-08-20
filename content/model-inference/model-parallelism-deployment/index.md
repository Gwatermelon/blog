---
title: "大模型推理并行部署：DP、TP、PP、EP、SP 与 CP 的区别"
date: 2026-08-20
lastmod: 2026-08-20
draft: false
description: "从模型、矩阵、层、MoE 专家和序列长度五个维度，梳理大模型推理部署中 DP、TP、PP、EP、SP 与 CP 的拆分方式和通信特点。"
summary: "DP 复制模型拆请求，TP 拆一层里的矩阵，PP 拆模型层，EP 拆 MoE 专家，SP/CP 则围绕序列维度和长上下文拆分 activation 与 attention。"
tags: ["模型推理", "并行部署", "DP", "TP", "PP", "EP", "SP", "CP", "MoE"]
categories: ["模型推理"]
math: true
ShowToc: true
TocOpen: true
---

大模型推理部署里的并行方式，经常可以先用一句话区分：

- **DP**：复制模型，拆请求；
- **TP**：拆一层里的矩阵；
- **PP**：拆模型的层；
- **EP**：拆 MoE 专家；
- **SP**：拆序列长度。

这些并行策略解决的问题并不相同。有的主要提升吞吐，有的解决单卡放不下模型的问题，有的降低 activation memory，有的则是 MoE 架构天然需要的专家分布方式。

## DP：Data Parallelism，数据并行

DP 是最简单的并行方式。它的做法是让每张 GPU 都保存一份完整模型，然后把不同请求分配给不同 GPU：

```text
GPU0：完整模型
GPU1：完整模型
GPU2：完整模型
GPU3：完整模型
```

请求分发时可以理解为：

```text
请求 A -> GPU0
请求 B -> GPU1
请求 C -> GPU2
请求 D -> GPU3
```

所以 DP 的核心是：**模型不拆，数据拆**。

比如部署一个 7B 模型，一张 A100 就能放下。现在同时来了 100 个请求，可以启动 4 份模型：

```text
┌─ GPU0：Model -> 一部分请求
Requests ──┼─ GPU1：Model -> 一部分请求
├─ GPU2：Model -> 一部分请求
└─ GPU3：Model -> 一部分请求
```

每张 GPU 都有完整权重，这就是 DP。

DP 主要提升的是吞吐量。例如：

```text
1 GPU：1000 token/s

4 GPU DP：
理论上接近 4000 token/s
```

但单个请求不会因为 `DP=4` 就快 4 倍，因为一个请求仍然只在一张 GPU 上执行。

DP 最大的问题是：它需要每张卡都保存完整模型。例如模型权重是 70 GB，那么：

```text
GPU0：70 GB
GPU1：70 GB
GPU2：70 GB
GPU3：70 GB

总共需要：280 GB
```

所以如果模型单卡根本放不下，纯 DP 没法解决。

## TP：Tensor Parallelism，张量并行

TP 是大模型推理中最核心的并行方式之一。它把一个大矩阵拆到多张 GPU 上。

为什么叫 Tensor Parallel？因为 Transformer 里大量计算本质上都是矩阵乘法：

$$
Y=XW
$$

比如：

```text
X: [B, S, 8192]
W: [8192, 32768]
```

如果 $W$ 很大，可以把它拆开。例如 `TP=4`：

```text
W = [ W0 | W1 | W2 | W3 ]
```

分别放在：

```text
GPU0 -> W0
GPU1 -> W1
GPU2 -> W2
GPU3 -> W3
```

于是每张 GPU 计算自己那部分矩阵乘法：

```text
GPU0: X x W0
GPU1: X x W1
GPU2: X x W2
GPU3: X x W3
```

最后把结果拼起来，以实现一个 Layer 内部多张 GPU 一起算。在切割过程中，本质上是在做向量的并行计算，所以叫 TP。

TP 会带来大量卡间通信。计算和通信高度耦合，是 TP 的特点。TP 通常要求 GPU 间有高速互联，比如：

```text
NVLink
NVSwitch
```

跨机器 TP 往往代价比较高。

## PP：Pipeline Parallelism，流水线并行

PP 的思路和 TP 完全不同。一句话解释：**TP 是横着切模型，PP 是竖着切模型**。

假设模型有 80 层，`PP=4`：

```text
GPU0：Layer 0~19
GPU1：Layer 20~39
GPU2：Layer 40~59
GPU3：Layer 60~79
```

一个请求的流动过程是：

```text
Input
↓
GPU0
Layer 0~19
↓
GPU1
Layer 20~39
↓
GPU2
Layer 40~59
↓
GPU3
Layer 60~79
↓
Output
```

这就是 Pipeline。这种实现可以让多个请求像流水线一样工作。

例如：

```text
时间 T1:
GPU0 -> Request A

时间 T2:
GPU0 -> Request B
GPU1 -> Request A

时间 T3:
GPU0 -> Request C
GPU1 -> Request B
GPU2 -> Request A

时间 T4:
GPU0 -> Request D
GPU1 -> Request C
GPU2 -> Request B
GPU3 -> Request A
```

它就像工厂流水线：

```text
工位1 -> 工位2 -> 工位3 -> 工位4
```

## PP 最大的问题：Pipeline Bubble

PP 的最大问题是流水线气泡。

假设刚开始：

```text
GPU0：工作
GPU1：等
GPU2：等
GPU3：等
```

结束阶段：

```text
GPU0：等
GPU1：等
GPU2：等
GPU3：工作
```

这些空闲时间叫 Pipeline Bubble，也就是流水线气泡。

所以 PP 想提高利用率，通常要多个 batch、多个请求或 micro batch，让流水线填满。

## TP 和 PP 的区别

假设有一个 80 层模型。

`TP=4` 时，每一层都拆成 4 份：

```text
Layer 0 -> GPU0 GPU1 GPU2 GPU3
Layer 1 -> GPU0 GPU1 GPU2 GPU3
Layer 2 -> GPU0 GPU1 GPU2 GPU3
...
```

四张 GPU 在每一层都一起工作。

`PP=4` 时：

```text
GPU0 -> Layer 0~19
GPU1 -> Layer 20~39
GPU2 -> Layer 40~59
GPU3 -> Layer 60~79
```

不同 GPU 负责不同的层。

## EP：Expert Parallelism，专家并行

EP 是把不同 MoE Expert 放到不同 GPU 上。

假设一个 MoE 模型有 256 个 experts，有 8 张 GPU，`EP=8`，那么每张 GPU 大约保存：

```text
256 / 8 = 32
```

个 Expert。例如：

```text
GPU0：Expert 0~31
GPU1：Expert 32~63
GPU2：Expert 64~95
...
GPU7：Expert 224~255
```

MoE 特别适合 EP 这种排布方式。

普通 FFN 的路径是：

```text
token
↓
FFN
↓
output
```

MoE 的路径是：

```text
token
↓
Router
↓
选择 Top-K experts
↓
Expert
↓
output
```

比如一个 token：

```text
Token A
```

Router 选择：

```text
Expert 7
Expert 123
```

假设：

```text
Expert 7   在 GPU0
Expert 123 在 GPU3
```

那么这个 token 的数据就需要发送到 GPU0 和 GPU3，算完再发送回来。

## EP 最核心的通信：All-to-All

例如，GPU0 当前有：

```text
token A -> Expert 70
token B -> Expert 130
token C -> Expert 2
```

但是：

```text
Expert 70  在 GPU2
Expert 130 在 GPU4
Expert 2   在 GPU0
```

那么就需要：

```text
GPU0 --token A--> GPU2
GPU0 --token B--> GPU4
```

其他 GPU 同样也在互相发送，于是形成：

```text
GPU0 <-> GPU1
GPU0 <-> GPU2
GPU0 <-> GPU3
...
GPU7 <-> GPU6
```

这种通信就是 All-to-All。

所以 MoE 推理里面经常看到：

```text
dispatch
all_to_all
expert compute
all_to_all
combine
```

完整流程可以理解成：

```text
token
↓
Router
↓
TopK expert
↓
Dispatch
↓
All-to-All
↓
Expert GEMM
↓
All-to-All
↓
Combine
```

## TP 和 EP 的区别

TP 和 EP 非常容易混淆。

假设 Expert 0 是一个 FFN。

EP 拆的是 Expert，例如：

```text
GPU0：Expert 0~31
GPU1：Expert 32~63
```

TP 拆的是 Expert 里面的矩阵：

```text
Expert 0:

W1
↓
拆成 4 块

GPU0 -> W1_0
GPU1 -> W1_1
GPU2 -> W1_2
GPU3 -> W1_3
```

所以：

```text
EP = 专家之间拆
TP = 一个专家内部再拆
```

它们甚至可以一起用。

## SP：Sequence Parallelism，序列并行

SP 是指沿着 sequence 或 token 维度拆 hidden states。用户输入的 token 数就是 sequence length。`SP=10`，可以理解为对 `sequence length / 10` 进行拆分。

假设：

```text
Sequence Length = 8192
```

有 4 张 GPU，`SP=4`：

```text
GPU0：token 0~2047
GPU1：token 2048~4095
GPU2：token 4096~6143
GPU3：token 6144~8191
```

Transformer 内经常看到：

```text
[B, S, H]
```

例如：

```text
[1, 8192, 4096]
```

其中：

```text
B = Batch Size
S = Sequence Length
H = Hidden Size
```

SP 拆的是 `S`。

原始 hidden states：

```text
[1, 8192, 4096]
```

`SP=4` 后，每张卡可能保存：

```text
GPU0: [1, 2048, 4096]
GPU1: [1, 2048, 4096]
GPU2: [1, 2048, 4096]
GPU3: [1, 2048, 4096]
```

这样拆分的主要目的是减少 activation memory，特别是 Prefill 阶段。

比如：

```text
Sequence = 128K
Hidden = 8192
```

hidden states 很大。如果所有 GPU 都保存完整 sequence：

```text
每卡：[128K, 8192]
```

会非常占内存。如果 `SP=8`：

```text
每卡：[16K, 8192]
```

内存压力会明显下降。

## SP 不能粗暴切开 Attention

对于 SP 来说，如果只是粗暴地把序列切成几段，然后每张 GPU 独立做 Attention，结果就是错的，因为 Attention 本来要求一个 token 能看到它应该看到的所有历史 token。

所以真正的 SP 并不是“每张卡只看自己的 token 然后各算各的”，而是：

> 在某些阶段按 Sequence 维度分片来节省 activation memory，但到了需要跨 token 交互的算子，尤其 Attention 时，通过通信把必要的信息补齐。

比方说做 LayerNorm。这些部分不需要 token-token interaction，因此非常容易沿 sequence 分：

```text
GPU0:
token 0~2047

GPU1:
token 2048~4095
```

每张卡单独做：

```text
LayerNorm
```

完全没有问题，因为 LayerNorm 一般是在 hidden dimension $H$ 上做：

```text
token0 -> 自己的 4096 个 hidden 做 norm
token1 -> 自己的 4096 个 hidden 做 norm
```

token0 不需要知道 token1，所以这里 SP 很自然。

## CP：Context Parallelism，面向长上下文的序列拆分

在涉及注意力机制的计算时，现在更多会指 Context Parallelism，也就是 CP。

例如 128K context：

```text
GPU0: token 0~31K
GPU1: token 32K~63K
GPU2: token 64K~95K
GPU3: token 96K~127K
```

Attention 又必须是全局 Attention，那么就需要：

```text
AllGather K/V
```

或者：

```text
Ring Attention
```

或者其他 distributed attention algorithm。

可以这样理解 SP 和 CP 的对比：

```text
Sequence Parallelism

主要目标：
拆 activation

常见场景：
LayerNorm / Residual / Dropout
和 TP 配合
```

```text
Context Parallelism

主要目标：
拆超长 sequence

尤其解决：
Attention 本身的 sequence 分片
```

## 总结

DP、TP、PP、EP、SP/CP 的区别可以收束成五种拆分维度：

| 并行方式 | 拆什么 | 典型作用 |
| --- | --- | --- |
| DP | 请求 / 数据 | 多副本提升吞吐 |
| TP | 一层里的矩阵 | 多卡共同计算一个 Layer |
| PP | 模型层 | 把不同层放到不同 GPU |
| EP | MoE Expert | 把不同 Expert 放到不同 GPU |
| SP | Sequence / token 维度 | 降低 activation memory |
| CP | 长上下文的 sequence 分片 | 处理 Attention 本身的序列拆分 |

其中 DP 不拆模型，所以单个请求不会因为 DP 增大而直接变快；TP 拆矩阵，计算和通信高度耦合；PP 拆层，会遇到 Pipeline Bubble；EP 拆专家，核心通信是 All-to-All；SP/CP 则围绕 sequence 维度，分别服务于 activation memory 和长上下文 Attention 分片。
