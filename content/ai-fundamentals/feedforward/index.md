---
title: "FeedForward"
date: 2026-07-21
lastmod: 2026-07-21
draft: false
description: "从激活函数和前馈网络出发，理解 Transformer FFN 的升维、非线性变换、逐 token 计算，以及现代大模型中的 SwiGLU 门控结构。"
summary: "梳理 Feed Forward 的基本含义、Transformer 中 FFN 与 Attention 的分工，并从 ReLU、SiLU、GLU 逐步推导到 SwiGLU。"
tags: ["AI 基础", "Transformer", "Feed Forward", "FFN", "SwiGLU"]
categories: ["AI 基础原理"]
math: true
ShowToc: true
TocOpen: true
---

Feed Forward 直译为“前馈”，核心含义是信息按照固定方向从输入流向输出，不存在循环反馈。在 Transformer 中，Feed Forward Network（FFN）是每个 Block 内部的重要子模块：Attention 负责不同 token 之间的信息交互，FFN 则独立地对每个 token 的特征进行非线性加工。

要理解 FFN，首先要从激活函数以及它所引入的非线性讲起。

## 激活函数

激活函数是一类数学函数，用来给神经网络引入非线性。神经网络中的一个线性层通常写成：

$$
y=Wx+b,
$$

其中 $x$ 是输入向量，$W$ 是权重矩阵，$b$ 是偏置，$y$ 是输出向量。

如果连续堆叠两个线性层：

$$
\begin{aligned}
h&=W_1x+b_1,\\
y&=W_2h+b_2,
\end{aligned}
$$

代入可得：

$$
y=W_2(W_1x+b_1)+b_2
  =(W_2W_1)x+(W_2b_1+b_2).
$$

令：

$$
W'=W_2W_1,\qquad b'=W_2b_1+b_2,
$$

那么：

$$
y=W'x+b'.
$$

也就是说，两层线性层本质上仍然等价于一层线性层。即使堆叠几十层，只要中间没有非线性操作，整个网络最终仍然只是一个线性变换，表达能力非常有限。

因此需要在两个线性层之间加入激活函数：

$$
h=\sigma(W_1x+b_1),
$$

这里的 $\sigma$ 就是激活函数。因为激活函数通常是非线性的，所以整个计算不再能合并成一次简单的矩阵乘法。

### 为什么需要非线性

假设输入只有一个数字 $x$，纯线性模型只能表示：

$$
y=ax+b.
$$

它只能画出一条直线，但现实中的关系通常不是线性的。例如：

- 图片中某些边缘组合起来才表示眼睛；
- 眼睛、鼻子和嘴巴组合起来才表示人脸；
- 某些 token 的组合才能表达否定、因果、讽刺等语义；
- 某个特征超过阈值后才应该激活。

激活函数使网络能够表示弯曲、分段、组合和条件性的复杂关系。可以把它理解为：线性层负责重新组合特征，激活函数负责决定哪些特征需要保留、增强或抑制。

## ReLU

ReLU 的全称是 Rectified Linear Unit，中文通常叫“修正线性单元”。其定义为：

$$
\operatorname{ReLU}(x)=\max(0,x)=\begin{cases}
x,&x>0,\\
0,&x\le 0.
\end{cases}
$$

因此 ReLU 所做的事情很简单：负数截断为 0，正数保持不变。

| 输入 $x$ | ReLU 输出 |
| ---: | ---: |
| $-3$ | $0$ |
| $-0.5$ | $0$ |
| $0$ | $0$ |
| $2$ | $2$ |
| $5$ | $5$ |

### ReLU 为什么能引入非线性

虽然 ReLU 在正数区域和负数区域分别都是线性的，但它整体不是一个线性函数。线性函数需要满足：

$$
f(x_1+x_2)=f(x_1)+f(x_2).
$$

取 $x_1=1,\ x_2=-2$，那么：

$$
\operatorname{ReLU}(x_1+x_2)
=\operatorname{ReLU}(-1)=0,
$$

但：

$$
\operatorname{ReLU}(x_1)+\operatorname{ReLU}(x_2)
=1+0=1.
$$

两者不相等，所以 ReLU 不是线性函数。神经网络通过大量线性层和 ReLU 的组合，可以构造非常复杂的分段线性函数。

### ReLU 的优点

1. **计算简单。** ReLU 只需要计算 $\max(0,x)$。相比 Sigmoid、Tanh 中的指数运算，计算成本较低。
2. **正半轴梯度稳定。** 当 $x>0$ 时：

$$
\operatorname{ReLU}'(x)=1.
$$

梯度不会因为激活函数本身不断缩小，这比 Sigmoid 在饱和区域中的梯度衰减问题要轻。

3. **激活具有稀疏性。** 负数会全部变成 0：

$$
[-2,3,-1,5]\longrightarrow[0,3,0,5].
$$

这种稀疏激活有时有利于特征选择和训练。

### ReLU 的问题：死亡 ReLU

由于负数区域的导数为 0，如果一个神经元长期满足：

$$
Wx+b<0,
$$

那么它的输出一直是 0，梯度也一直是 0，权重可能再也无法更新回来。这种情况称为 Dying ReLU，即“死亡 ReLU”。

为缓解这个问题，可以使用 Leaky ReLU：

$$
\operatorname{LeakyReLU}(x)=\begin{cases}
x,&x>0,\\
\alpha x,&x\le 0,
\end{cases}
$$

其中 $\alpha$ 是一个很小的正数，例如 $0.01$。这样负数区域仍然保留一个很小的梯度：

$$
\operatorname{LeakyReLU}'(x)=\alpha,\qquad x<0.
$$

## Feed Forward 的含义

“Feed Forward”直译是“前馈”。它表示信息按照固定方向从输入流向输出，不存在循环反馈。例如：

$$
x\longrightarrow\operatorname{Linear}
\longrightarrow\operatorname{Activation}
\longrightarrow\operatorname{Linear}
\longrightarrow y.
$$

这是一个前馈计算过程。

### 广义的前馈神经网络

最普通的多层感知机 MLP，就是一种前馈神经网络：

$$
x\rightarrow\operatorname{Linear}_1
\rightarrow\operatorname{ReLU}
\rightarrow\operatorname{Linear}_2
\rightarrow\operatorname{ReLU}
\rightarrow\operatorname{Linear}_3
\rightarrow y.
$$

它和循环神经网络 RNN 的区别在于：

- 前馈网络中，信息只向前传播；
- RNN 中，前一时刻的隐藏状态会反馈给下一时刻。

不过，训练前馈网络时仍然会进行反向传播。“前馈”描述的是模型的计算图结构，并不表示训练时没有反向传播。

## Transformer 中的 Feed Forward Network

在 Transformer 中，FFN 通常是每个 Transformer Block 内部的一个子模块。经典形式为：

$$
\operatorname{FFN}(x)
=W_2\operatorname{ReLU}(W_1x+b_1)+b_2.
$$

也可以写成：

$$
x\xrightarrow{W_1}h
\xrightarrow{\operatorname{ReLU}}a
\xrightarrow{W_2}y.
$$

其中通常先升维，再降维：

$$
d_{\text{model}}\rightarrow d_{\text{ff}}\rightarrow d_{\text{model}}.
$$

例如：

$$
4096\rightarrow16384\rightarrow4096.
$$

若输入 token 向量为 $x\in\mathbb{R}^{4096}$，经过第一个线性层：

$$
h=W_1x+b_1,\qquad h\in\mathbb{R}^{16384},
$$

再经过激活函数：

$$
a=\operatorname{ReLU}(h),
$$

最后通过第二个线性层恢复原始隐藏维度：

$$
y=W_2a+b_2,\qquad y\in\mathbb{R}^{4096}.
$$

因此，FFN 不是一个激活函数，而是一个包含线性投影和激活函数的完整模块。

![Transformer 模型中的 Feed Forward 子层](transformer-architecture.png)

### FFN 为什么要先升维再降维

假设输入向量是 4 维：

$$
x=[x_1,x_2,x_3,x_4].
$$

第一个线性层把它投影到更大的特征空间：

$$
4\rightarrow16.
$$

这相当于生成更多中间特征：

$$
h=[h_1,h_2,\ldots,h_{16}],
$$

其中每个 $h_i$ 都是输入各维度的不同组合。激活函数再对这些中间特征进行非线性筛选：

$$
a_i=\operatorname{ReLU}(h_i).
$$

第二个线性层把这些中间特征重新组合，并降回原始隐藏维度：

$$
16\rightarrow4.
$$

可以粗略理解为：

1. 第一个线性层展开出大量候选特征；
2. 激活函数筛选或门控这些特征；
3. 第二个线性层把特征组合回原始隐藏维度。

升维提供了更大的中间表示空间，使 FFN 能学习更复杂的变换。

### Attention 和 FFN 的分工

Transformer Block 可以简化为：

$$
x\longrightarrow\operatorname{Attention}
\longrightarrow\operatorname{FFN}.
$$

二者作用不同：

- **Attention** 主要负责不同 token 之间的信息交互；
- **FFN** 主要负责对每个 token 自身的特征进行非线性加工。

同一个 FFN 会独立应用到每个 token：

$$
y_i=\operatorname{FFN}(x_i).
$$

不同 token 之间不会在 FFN 内直接交互。换句话说，Attention 在 token 维度上混合信息，而 FFN 在特征维度上变换信息。

### 一个具体的 FFN 计算示例

假设输入和第一层权重为：

$$
x=
\begin{bmatrix}
1\\
2
\end{bmatrix},
\qquad
W_1=
\begin{bmatrix}
1&1\\
-1&1\\
2&-1
\end{bmatrix}.
$$

暂时忽略偏置，第一个线性层得到：

$$
h=W_1x=\begin{bmatrix}
1\times1+1\times2\\
-1\times1+1\times2\\
2\times1-1\times2
\end{bmatrix}=\begin{bmatrix}
3\\
1\\
0
\end{bmatrix}.
$$

经过 ReLU 后 $a=[3,1,0]^{\mathsf T}$。假设第二个线性层为：

$$
W_2=
\begin{bmatrix}
1&2&1\\
-1&1&2
\end{bmatrix},
$$

那么：

$$
y=W_2a=\begin{bmatrix}
1\times3+2\times1+1\times0\\
-1\times3+1\times1+2\times0
\end{bmatrix}=\begin{bmatrix}
5\\
-2
\end{bmatrix}.
$$

整个过程完成了：

$$
\mathbb{R}^{2}\rightarrow\mathbb{R}^{3}
\xrightarrow{\operatorname{ReLU}}\mathbb{R}^{3}
\rightarrow\mathbb{R}^{2}.
$$

## 现代大模型中的激活与门控

经典 Transformer 论文中的 FFN 使用 ReLU，但现代大语言模型更常使用 GELU、SiLU/Swish、GLU、SwiGLU 或 GeGLU。其中，SwiGLU 是现代 Transformer 中常见的一种门控前馈网络结构。

SwiGLU 不是单纯的激活函数，而是：

> Swish/SiLU 激活函数 + GLU 门控机制。

它通常用来替代经典的：

$$
\operatorname{Linear}\rightarrow\operatorname{ReLU}\rightarrow\operatorname{Linear}.
$$

### SiLU

SwiGLU 中的 “Swi” 来自 Swish。实践中通常使用的 Swish 形式就是 SiLU：

$$
\operatorname{SiLU}(x)=x\cdot\sigma(x),
\qquad
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

所以：

$$
\operatorname{SiLU}(x)=\frac{x}{1+e^{-x}}.
$$

和 ReLU 相比，ReLU 会直接将负数截断为 0，而 SiLU 会保留一个较小的负值。因此：

- ReLU 在 $x=0$ 处有折点；
- SiLU 处处平滑、可导；
- SiLU 的负半轴仍然可以传播梯度；
- SiLU 不容易出现严重的“死亡神经元”。

可以粗略理解为：

$$
\operatorname{SiLU}(x)=x\times\text{软门控系数},
$$

因为 $\sigma(x)\in(0,1)$，Sigmoid 决定输入 $x$ 被保留多少。

### GLU

GLU 的全称是 Gated Linear Unit，即门控线性单元。普通 FFN 只有一条升维分支：

$$
h=xW_{\text{up}}.
$$

GLU 会把输入投影成两条分支：

$$
a=xW_{\text{gate}},\qquad b=xW_{\text{up}},
$$

然后一条分支充当“门”，另一条分支充当“内容”：

$$
h=\sigma(a)\odot b,
$$

其中 $\sigma(a)$ 决定内容分支的每个维度保留多少，$\odot$ 表示对应元素相乘。

例如：

$$
\sigma(a)=[0.1,0.9,0.3],\qquad b=[10,5,-2],
$$

那么：

$$
h=[0.1,0.9,0.3]\odot[10,5,-2]=[1,4.5,-0.6].
$$

第一个维度被大幅抑制，第二个维度基本被保留，第三个维度被部分保留。门控结构让网络不仅能决定“激活值是多少”，还可以根据输入动态决定“这条特征通道应该打开多少”。

### SwiGLU 的计算

SwiGLU 把 SiLU 激活与 GLU 门控结合起来。其核心计算为：

$$
h=\operatorname{SiLU}(xW_{\text{gate}})
\odot(xW_{\text{up}}),
$$

然后通过降维矩阵：

$$
y=hW_{\text{down}}.
$$

合起来就是：

$$
y=
\left[
\operatorname{SiLU}(xW_{\text{gate}})
\odot(xW_{\text{up}})
\right]W_{\text{down}}.
$$

其中 $W_{\text{gate}}$ 生成门控分支，$W_{\text{up}}$ 生成内容分支，$W_{\text{down}}$ 将结果降回隐藏维度。虽然实现比 ReLU FFN 更复杂，但本质仍然是：线性投影 $\rightarrow$ 非线性激活或门控 $\rightarrow$ 线性投影。

### SwiGLU 的优势

1. **更强的动态特征选择。** 门控分支会根据当前输入动态调制内容分支，不同 token 会生成不同的门控值；
2. **激活函数平滑。** SiLU 没有 ReLU 在 0 处的尖锐折点，梯度变化更加平滑；
3. **负半轴保留梯度。** SiLU 在大部分负半轴仍有非零梯度，因此不容易完全失活；
4. **引入乘法交互。** 两条输入相关分支之间进行逐元素乘法：

$$
f(x)\odot g(x),
$$

这种结构比单分支激活拥有更丰富的函数表达能力。在参数量和计算量相近的条件下，门控 FFN 往往比经典 ReLU/GELU FFN 表现更好，因此现代大语言模型广泛采用类似结构。

### SwiGLU 的代价

普通 FFN 有两个矩阵：

$$
W_1,\quad W_2.
$$

SwiGLU 有三个矩阵：

$$
W_{\text{gate}},\quad W_{\text{up}},\quad W_{\text{down}}.
$$

因此如果中间维度不缩小，计算量和参数量会显著增加。SwiGLU 还需要同时产生：

$$
G=XW_{\text{gate}},
\qquad
U=XW_{\text{up}},
$$

这会增加中间激活显存和内存带宽压力。推理框架通常会尝试融合 gate projection、up projection、SiLU 和 elementwise multiply，否则可能产生更多 kernel launch 和中间张量读写。

## 总结

Feed Forward 的核心不是简单地“向前计算”，而是通过线性投影和非线性激活，对特征进行逐 token 加工：

1. 没有激活函数时，多层线性层仍可合并为一层，表达能力有限；
2. ReLU 计算简单、正半轴梯度稳定，但存在死亡 ReLU 问题；
3. Transformer FFN 通常先升维、执行非线性变换，再降回隐藏维度；
4. Attention 负责 token 间的信息混合，FFN 负责每个 token 内部的特征变换；
5. SwiGLU 通过 SiLU 和门控乘法提供更强的动态特征选择，但会增加投影、显存和实现成本。

## 参考

1. [神经网络中的激活函数与前馈网络](https://blog.csdn.net/weixin_51756104/article/details/127250190)
