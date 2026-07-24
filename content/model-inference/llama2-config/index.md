---
title: "从 config.json 读懂 Llama 2 7B"
date: 2026-07-24
lastmod: 2026-07-24
draft: false
description: "以 Llama 2 7B 的 config.json 为入口，还原模型架构、注意力类型、RoPE、RMSNorm、SwiGLU、KV Cache、显存占用与参数量。"
summary: "config.json 不只是模型加载配置：从层数、隐藏维度、注意力头、上下文长度和精度字段，可以还原 Llama 2 7B 的网络结构并估算推理资源。"
tags: ["模型推理", "Llama 2", "config.json", "KV Cache", "LLM Inference"]
categories: ["模型推理"]
math: true
ShowToc: true
TocOpen: true
---

拿到一个大语言模型后，最快的认识方式往往不是立刻阅读全部源码，而是先打开它的 `config.json`。这个文件可以看作模型的**结构说明书**：它描述模型有多少层、每层多宽、注意力头如何划分、词表多大，以及使用什么归一化和位置编码。

本文以 [ModelScope 上的 Llama 2 7B 配置](https://modelscope.cn/models/LLM-Research/llama-2-7b/files) 为例，说明如何从配置还原网络结构，并进一步判断推理阶段的计算量、KV Cache 和显存需求。

## config.json 不包含模型权重

先区分两个容易混淆的概念：

```text
config.json        → 按配置创建模型骨架
model.safetensors  → 把训练好的参数填入模型骨架
```

权重也可能被拆成多个分片，例如：

```text
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
```

因此，只有 `config.json` 不能运行模型，但它足以回答很多结构问题。本文使用的关键字段如下：

```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 32,
  "max_position_embeddings": 4096,
  "hidden_act": "silu",
  "attention_bias": false,
  "attention_dropout": 0.0,
  "mlp_bias": false,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "rope_scaling": null,
  "vocab_size": 32000,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "use_cache": true
}
```

一句话概括，这是一套由 32 层 Decoder Block 组成、隐藏维度为 4096、使用标准多头注意力、RoPE、RMSNorm 和 SwiGLU 风格 MLP 的 Decoder-only 因果语言模型。

## 从输入到输出的完整数据流

Llama 2 7B 的整体数据流可以写成：

```text
输入文本
  ↓
Tokenizer
  ↓
token IDs：[B, T]
  ↓
Token Embedding
  ↓
隐藏状态：[B, T, 4096]
  ↓
Llama Decoder Layer × 32
  ↓
Final RMSNorm
  ↓
LM Head：4096 → 32000
  ↓
每个位置对下一个 token 的 logits
```

其中：

- $B$ 是 batch size；
- $T$ 是序列长度；
- $4096$ 是每个 token 的隐藏向量维度。

例如，`[2, 100, 4096]` 表示一个 batch 中有 2 条序列，每条序列包含 100 个 token，每个 token 用 4096 个数表示。这里的 4096 是隐藏维度，不是 token 数量。

每个 Decoder Block 的结构为：

```text
输入 x
  ├─ RMSNorm → Causal Self-Attention → Residual Add
  └─ RMSNorm → SwiGLU MLP            → Residual Add
```

这是一种 Pre-Norm 结构：先做归一化，再进入注意力或 MLP，最后与残差相加。

## 用 architectures 判断模型用途

配置中的

```json
"architectures": ["LlamaForCausalLM"]
```

表示 Hugging Face Transformers 加载模型时应创建 `LlamaForCausalLM`。这个名称可以拆成两部分：

- `Llama`：使用 Llama 系列的网络结构；
- `ForCausalLM`：增加因果语言建模所需的 LM Head。

代码关系可以粗略理解为：

```python
class LlamaForCausalLM:
    def __init__(self, config):
        self.model = LlamaModel(config)
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
```

`LlamaForCausalLM` 只是代码模板，具体实例有多大、使用多少个注意力头，仍然由配置决定。同一个类可以根据不同配置创建 Llama 2 7B、Llama 2 70B 或其他 Llama 系列模型。

`"model_type": "llama"` 则用于让 Transformers 识别配置所属的模型家族。

## 层数、隐藏维度与 MLP 宽度

### num_hidden_layers

```json
"num_hidden_layers": 32
```

表示模型包含 32 个 Transformer Decoder Block。通常所说的“32 层”不包括：

- 输入 Embedding；
- 最后的 RMSNorm；
- LM Head。

### hidden_size

```json
"hidden_size": 4096
```

`hidden_size` 也常写作 $d_{\text{model}}$。每个 token 在模型内部都由一个 4096 维向量表示：

$$
X\in\mathbb{R}^{B\times T\times 4096}.
$$

这个维度同时决定注意力投影、残差流、归一化参数和 LM Head 输入的宽度，是模型规模的核心参数之一。

### intermediate_size

```json
"intermediate_size": 11008
```

它表示 MLP 的中间维度。隐藏状态会先从 4096 维扩展到 11008 维，再投影回 4096 维。

Llama 使用的不是简单的 `Linear → ReLU → Linear`，而是带门控的 SwiGLU 风格结构，包含三组矩阵：

```text
gate_proj：4096  → 11008
up_proj：  4096  → 11008
down_proj：11008 → 4096
```

其计算可以写成：

$$
\operatorname{MLP}(x)
=W_{\text{down}}
\left[
\operatorname{SiLU}(W_{\text{gate}}x)
\odot
(W_{\text{up}}x)
\right].
$$

`"hidden_act": "silu"` 指定门控分支使用 SiLU，`"mlp_bias": false` 表示三个线性层均不使用偏置。

## 从 Q 头和 KV 头判断注意力类型

配置中最值得关注的一组字段是：

```json
"num_attention_heads": 32,
"num_key_value_heads": 32
```

前者表示 Query 有 32 个头，后者表示 Key、Value 各有 32 个头。单个头的维度为：

$$
d_{\text{head}}
=\frac{d_{\text{model}}}{n_{\text{heads}}}
=\frac{4096}{32}
=128.
$$

输入隐藏状态经过投影后，张量形状从

$$
[B,T,4096]
$$

变为

$$
Q,K,V\in\mathbb{R}^{B\times 32\times T\times 128}.
$$

每个头独立计算：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{128}}
\right)V.
$$

32 个头的输出拼接后重新回到 4096 维。

### MHA、GQA 与 MQA

可以通过 Query 头数与 KV 头数的关系判断注意力类型：

| 类型 | 头数关系 | 含义 |
| --- | --- | --- |
| MHA | $n_q=n_{kv}$ | 每个 Query 头有独立的 KV 头 |
| GQA | $1\lt n_{kv}\lt n_q$ | 多个 Query 头共享一组 KV 头 |
| MQA | $n_{kv}=1$ | 所有 Query 头共享一组 KV 头 |

这份 Llama 2 7B 配置中：

$$
n_q=n_{kv}=32,
$$

因此使用的是标准 Multi-Head Attention，而不是 GQA。不能因为 Llama 2 70B 使用 GQA，就推断所有 Llama 2 规格都使用 GQA，最终仍应检查 `num_key_value_heads`。

对于推理而言，KV 头数量直接影响 KV Cache 大小。其他条件相同时，KV 头从 32 减少到 8，缓存理论大小也会降到原来的四分之一。

## Attention 中的 bias 与 dropout

```json
"attention_bias": false,
"attention_dropout": 0.0
```

`attention_bias: false` 表示 $Q$、$K$、$V$ 和输出投影的线性层不使用 bias，例如：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

注意力权重可以记为：

$$
A=\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_{\text{head}}}}
\right).
$$

训练时，一些模型还会计算

$$
A'=\operatorname{Dropout}(A).
$$

这里的 Dropout 概率为 0，因此训练时也不会随机丢弃注意力权重。推理模式本来就会关闭 Dropout，所以这一字段主要影响训练行为。

## RoPE 与上下文长度

```json
"max_position_embeddings": 4096,
"rope_theta": 10000.0,
"rope_scaling": null
```

这三个字段应放在一起理解：

- `max_position_embeddings: 4096`：配置对应的原生最大位置范围为 4096；
- `rope_theta: 10000`：控制 RoPE 旋转频率的分布；
- `rope_scaling: null`：没有启用额外的 RoPE 缩放方案。

推理时，输入 prompt 与已经生成的 token 会共同占用上下文：

$$
T_{\text{context}}
=T_{\text{prompt}}+T_{\text{generated}}.
$$

如果配置启用了上下文扩展，可能会出现类似字段：

```json
"rope_scaling": {
  "type": "linear",
  "factor": 2.0
}
```

但扩展位置范围并不自动保证长上下文质量不下降，还需要与模型训练方式和推理实现共同评估。

## RMSNorm

```json
"rms_norm_eps": 1e-5
```

Llama 使用 RMSNorm，而不是原始 Transformer 中的 LayerNorm。对于 $d$ 维向量 $x$，先计算：

$$
\operatorname{RMS}(x)
=\sqrt{
\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon
}.
$$

再进行缩放：

$$
\operatorname{RMSNorm}(x)
=\frac{x}{\operatorname{RMS}(x)}\odot\gamma.
$$

在这份配置中：

$$
d=4096,\qquad \epsilon=10^{-5},
$$

$\gamma$ 是可训练的缩放参数。$\epsilon$ 用于避免分母为零或过于接近零，从而提高数值稳定性。

## 词表、特殊 token 与输出层

```json
"vocab_size": 32000,
"bos_token_id": 1,
"eos_token_id": 2,
"pad_token_id": 0
```

词表包含 32000 个 token。输入 Embedding 的权重形状为：

$$
W_{\text{embed}}\in\mathbb{R}^{32000\times4096},
$$

LM Head 把隐藏向量转换为词表 logits：

$$
W_{\text{lm}}\in\mathbb{R}^{32000\times4096}.
$$

特殊 token 的作用分别是：

- BOS：序列开始标记，ID 为 1；
- EOS：序列结束标记，ID 为 2；
- PAD：批处理补齐标记，ID 为 0。

例如两条不同长度的序列可以补齐为：

```text
A tokens：[11, 25, 37, 48, 52]
B tokens：[19, 21, 33,  0,  0]

A mask：  [ 1,  1,  1,  1,  1]
B mask：  [ 1,  1,  1,  0,  0]
```

Attention Mask 会让模型忽略补齐位置。

需要注意，这份配置的 `_name_or_path` 指向 `unsloth/llama-2-7b`，说明它经过 Unsloth 打包或处理。`pad_token_id` 等 tokenizer 适配字段未必与最初发布版本完全一致，部署时应以当前仓库配套的 tokenizer 文件为准。

## Embedding 与 LM Head 是否共享

```json
"tie_word_embeddings": false
```

这表示输入 Embedding 与输出 LM Head 不共享权重。两者各自包含：

$$
32000\times4096
=131{,}072{,}000
$$

个参数，合计约 2.62 亿。如果启用 Weight Tying，则可以省去其中一份独立参数。

## use_cache 与 KV Cache

```json
"use_cache": true
```

表示生成时默认返回并复用历史 Key、Value。若不使用 KV Cache，每生成一个新 token 都要重新计算完整前缀；启用缓存后，只需计算新 token 对应的 Query、Key、Value，并读取历史缓存。

对于普通 MHA，每层 KV Cache 的元素数量近似为：

$$
N_{\text{KV, layer}}
=2BTn_{kv}d_{\text{head}},
$$

其中系数 2 分别对应 Key 和 Value。全模型缓存字节数近似为：

$$
M_{\text{KV}}
=2BTLn_{kv}d_{\text{head}}s,
$$

其中 $L$ 是层数，$s$ 是每个元素的字节数。

以 batch size 为 1、上下文长度为 4096、BF16 为例：

$$
M_{\text{KV}}
=2\times1\times4096\times32\times32\times128\times2
=2{,}147{,}483{,}648\ \text{bytes},
$$

约为 2 GiB。这还没有包含权重、临时激活、CUDA Kernel 工作空间和框架管理开销。

## torch_dtype 与权重显存

```json
"torch_dtype": "bfloat16"
```

BF16 每个参数占 2 字节。对于约 67.4 亿参数：

$$
6.74\times10^9\times2
\approx13.48\times10^9\ \text{bytes}.
$$

也就是说，仅模型权重就约占 13.48 GB，折合约 12.55 GiB。实际推理显存还需要容纳：

- KV Cache；
- 临时激活；
- CUDA Kernel 工作空间；
- 推理框架的管理开销。

因此，“权重小于显存容量”不等于模型一定能够高效运行。`torch_dtype` 也只是默认加载提示，调用 `from_pretrained` 时仍可显式覆盖，或者加载量化权重。

## 从配置还原一层 Llama Decoder

把关键字段组合起来，一层 Decoder 可以近似写成：

```python
class LlamaDecoderLayer:
    def __init__(self):
        self.input_layernorm = RMSNorm(
            hidden_size=4096,
            eps=1e-5,
        )

        self.self_attn = {
            "q_proj": Linear(4096, 4096, bias=False),
            "k_proj": Linear(4096, 4096, bias=False),
            "v_proj": Linear(4096, 4096, bias=False),
            "o_proj": Linear(4096, 4096, bias=False),
            "num_heads": 32,
            "num_kv_heads": 32,
            "head_dim": 128,
            "rope_theta": 10000,
        }

        self.post_attention_layernorm = RMSNorm(
            hidden_size=4096,
            eps=1e-5,
        )

        self.mlp = {
            "gate_proj": Linear(4096, 11008, bias=False),
            "up_proj": Linear(4096, 11008, bias=False),
            "down_proj": Linear(11008, 4096, bias=False),
            "activation": SiLU(),
        }
```

前向计算可以概括为：

```python
# Attention
residual = x
x = rms_norm_1(x)
x = self_attention(x)
x = residual + x

# MLP
residual = x
x = rms_norm_2(x)
x = down_proj(silu(gate_proj(x)) * up_proj(x))
x = residual + x
```

这一层重复 32 次后，再经过 Final RMSNorm 和 LM Head。

## 从配置估算参数量

### 单层 Attention

$Q$、$K$、$V$、$O$ 四个投影矩阵共包含：

$$
4\times4096\times4096
=67{,}108{,}864
$$

个参数，约 6711 万。

### 单层 MLP

`gate_proj`、`up_proj`、`down_proj` 三个矩阵共包含：

$$
3\times4096\times11008
=135{,}266{,}304
$$

个参数，约 1.353 亿。

忽略少量 RMSNorm 参数，单个 Decoder Block 约有：

$$
67.11\ \text{M}+135.27\ \text{M}
\approx202.38\ \text{M}
$$

个参数。32 层约为：

$$
202.38\ \text{M}\times32
\approx6.476\ \text{B}.
$$

再加上不共享的 Embedding 与 LM Head：

$$
2\times32000\times4096
=0.262144\ \text{B}.
$$

计入 RMSNorm 后，总参数量约为：

$$
6.476\ \text{B}+0.262\ \text{B}
\approx6.74\ \text{B}.
$$

这也解释了为什么型号写作“7B”，而更精确的参数规模约为 6.74B。

## 哪些字段不直接决定网络结构

配置中还有一些运行信息或元数据：

- `transformers_version`：保存配置时使用或兼容的 Transformers 版本；
- `unsloth_version`：模型经过哪个版本的 Unsloth 工具处理；
- `_name_or_path`：模型保存前的名称或来源路径；
- `initializer_range`：随机初始化新参数时使用的分布范围；
- `pretraining_tp`：兼容预训练阶段张量并行切分的字段。

尤其要注意：

```json
"pretraining_tp": 1
```

不表示模型只能单卡运行，也不表示推理时 tensor parallel size 必须为 1。vLLM 等推理框架中的张量并行配置属于另一个层面。

## 阅读模型配置的推荐顺序

以后拿到一个陌生模型，可以按照下面的顺序阅读：

1. **判断模型用途**：查看 `architectures` 和 `model_type`。
2. **判断基础规模**：查看 `num_hidden_layers`、`hidden_size` 和 `intermediate_size`。
3. **判断注意力类型**：比较 `num_attention_heads` 与 `num_key_value_heads`。
4. **判断上下文能力**：查看 `max_position_embeddings`、`rope_theta` 和 `rope_scaling`。
5. **判断 FFN 与归一化**：查看 `hidden_act`、`intermediate_size` 和 `rms_norm_eps`，并结合代码确认具体实现。
6. **判断词表与输出层**：查看 `vocab_size`、特殊 token ID 和 `tie_word_embeddings`。
7. **估算推理资源**：结合 `torch_dtype`、`use_cache`、KV 头数、层数和上下文长度计算权重与 KV Cache。
8. **区分结构字段和元数据**：不要把 `transformers_version`、`_name_or_path` 或 `pretraining_tp` 误认为网络层配置。

## 总结

`config.json` 的价值不只是让框架能够加载模型。正确阅读这些字段后，可以从一个很小的配置文件还原出：

- 模型是 Decoder-only 因果语言模型；
- 主干包含 32 个 Decoder Block；
- 隐藏维度为 4096，MLP 中间维度为 11008；
- 32 个 Query 头和 32 个 KV 头构成标准 MHA；
- 单头维度为 128；
- 使用 RoPE、RMSNorm 和 SwiGLU 风格 MLP；
- 原生上下文长度为 4096；
- Embedding 与 LM Head 不共享；
- BF16 权重约为 13.48 GB；
- 4096 token、batch size 为 1 时，KV Cache 约为 2 GiB；
- 总参数量约为 6.74B。

从配置阅读模型，本质上是在建立一条完整链路：

```text
字段 → 张量形状 → 网络结构 → 参数量 → 推理显存与性能
```

掌握这条链路后，面对其他 Llama、Qwen 或 Hugging Face 模型，也可以用同样的方法快速定位其结构与推理特征。

