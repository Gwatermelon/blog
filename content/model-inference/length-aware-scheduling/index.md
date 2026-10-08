---
title: "从 Padding 到 Continuous Batching：LLM 推理中的长度感知调度"
date: 2026-10-08
lastmod: 2026-10-08
draft: false
description: "从 Length Bucketing、Token Budget、Packed/Ragged Batching 到 Continuous Batching 与 Chunked Prefill，梳理变长请求在 LLM 推理中的组批、存放和调度方式。"
summary: "长度感知调度不只是在组批前按长度分桶，还要同时控制请求组合、每轮 Token 预算、变长数据布局以及请求的动态进出。"
tags: ["模型推理", "LLM Inference", "Length Bucketing", "Continuous Batching", "Chunked Prefill", "Token Budget", "vLLM", "SGLang"]
categories: ["模型推理"]
math: true
ShowToc: true
TocOpen: true
---

Length Bucketing（长度分桶）是解决变长序列批处理时 Padding 无效计算的经典方法。但在今天的 LLM 推理系统中，它已经扩展成一整套长度感知调度机制：Token Budget、无 Padding 数据布局、迭代级调度和 Chunked Prefill 会共同决定吞吐、延迟与 GPU 利用率。

本文从模型已经部署、请求开始进入推理引擎的时刻出发，讨论三类问题：

| 问题 | 典型技术 | 关注点 |
| --- | --- | --- |
| 哪些请求一起计算？ | Length Bucketing、Dynamic Padding、Token-aware Batching | 降低 Padding，稳定每轮工作量 |
| 变长序列如何存放？ | Packed/Ragged Batching | 只保存并处理有效 Token |
| 请求何时进入和退出？ | Continuous Batching、Chunked Prefill | 动态补位，让长 Prompt 分轮进入 |

这些方法不是互相替代的方案，而是分别作用于组批、数据布局和运行时调度三个层次。

## 哪些请求一起计算？

### Padding 浪费如何产生

假设同一时刻到达 4 个请求，输入长度分别为 512、1,024、1,536 和 8,192。若规则矩阵要求全部序列补齐到本批次最长长度，则会得到：

| 请求 | 有效长度 | 补齐后长度 |
| --- | ---: | ---: |
| A | 512 | 8,192 |
| B | 1,024 | 8,192 |
| C | 1,536 | 8,192 |
| D | 8,192 | 8,192 |

此时张量形状为 `[4, 8192]`，总 Token 槽位数为：

$$
N_{\text{padded}}=4\times8192=32768
$$

真正有效的 Token 数为：

$$
N_{\text{useful}}=512+1024+1536+8192=11264
$$

因此，有效 Token 比例只有：

$$
\eta=\frac{11264}{32768}\times100\%=34.4\%
$$

其余约 $65.6\%$ 的位置都是 Padding。

Padding 的代价还不只取决于 Token 数。标准 Prefill Attention 的计算量近似与序列长度平方相关；若按规则矩阵计算，量级接近 $B\cdot L_{\max}^{2}$，而只计算有效序列时则接近 $\sum_i L_i^2$。在这个例子中，Attention 的有效计算比例近似为：

$$
\frac{512^2+1024^2+1536^2+8192^2}{4\times8192^2}\approx26.4\%
$$

投影层和 MLP 的计算量主要随有效 Token 数线性增长。因此，前面的 $34.4\%$ 只表示 Token 槽位利用率，不能代表全部算子的计算效率。同一批请求的长度差异过大，会同时影响计算效率、显存占用和延迟。

### Length Bucketing：让相近长度的请求一起算

Length Bucketing 的基本思路，是优先让输入长度接近的请求一起执行 Forward。例如：

```text
Incoming Requests
        │
        ▼
按 Input Length 分类
        │
        ├── Bucket A: 0～1K
        ├── Bucket B: 1K～2K
        ├── Bucket C: 2K～4K
        ├── Bucket D: 4K～8K
        └── Bucket E: >8K
```

这样可以缩小同一批次中最长序列与最短序列的差距，从而减少 Padding。不过，分桶越细并不一定越好：如果为了凑齐同长度请求而等待太久，队列延迟会抵消计算侧的收益。在线服务通常需要在 Padding 浪费和等待时间之间取平衡。

### Dynamic Padding：只补齐到当前批次最大长度

Static Padding 可能把每个输入都补齐到预设的全局长度。例如模型允许 8,192 Token，而当前批次长度只有 128、160 和 192：

```text
Static Padding
128 → 8192
160 → 8192
192 → 8192
```

Dynamic Padding 则只补齐到当前批次的最大长度：

```text
Dynamic Padding
128 → 192
160 → 192
192 → 192
```

它仍然保留 Padding，但显著缩小了补齐范围。在训练或离线处理场景中，常见做法是再配合 `group_by_length`，把长度接近的样本放在一起。

### Token-aware Batching：限制一轮实际处理的 Token

Length Bucketing 回答“谁和谁一起算”，Token-aware Batching 则回答“这一轮一共允许算多少”。

在 LLM 中，请求数不能准确代表一次 Forward 的计算负载：`batch_size=8` 既可能只包含 512 个 Token，也可能包含 64K Token。因此，现代推理引擎通常会同时限制：

- 本轮最多容纳多少个请求或序列；
- 本轮最多处理多少个 Token。

以 Token Budget 为 8,192、最大请求数为 32 为例：

| 单请求输入长度 | Token Budget 允许的请求数 | 本轮 Token 数 |
| ---: | ---: | ---: |
| 512 | 16 | 8,192 |
| 2,048 | 4 | 8,192 |
| 8,192 | 1 | 8,192 |

这样，Batch Size 会随输入长度动态变化，而每轮处理的 Token 数在理想情况下更接近设定预算，GPU 每次 Forward 的工作量也更稳定。不过，Token Budget 只是比请求数更好的容量约束，并不是精确的算力预算：相同 Token 数的 Prefill 与 Decode 具有不同的计算和访存特征，Prefill 成本也会受到序列长度分布影响。

这一点对短输入的 Prefill 尤其重要。假设输入只有 64 Token、预期输出为 4,096 Token：若 Prefill 每次固定只接收 8 个请求，一轮只有 $8\times64=512$ 个 Token，GEMM 规模可能不足以充分利用 GPU；若 Token Budget 为 4,096，则最多有机会把 64 个同类短请求放入同一轮，提升计算密度和 GPU 利用率。当然，实际可接收数量仍会受到最大请求数、KV Cache、显存和队列状态等条件限制。

在 vLLM 中，这两个维度分别对应 `max_num_seqs` 和 `max_num_batched_tokens`：前者限制单次迭代中的序列数，后者限制单次迭代调度的 Token 数。

## 变长序列如何存放？

### Packed/Ragged Batching

Length Bucketing 和 Dynamic Padding 都是在减少 Padding，Packed/Ragged Batching 则从数据布局上避免为 Padding 分配同等的计算和存储。

假设三个请求的长度分别为 3、5 和 2：

```text
Request A: [A1 A2 A3]
Request B: [B1 B2 B3 B4 B5]
Request C: [C1 C2]
```

规则矩阵需要处理 $3\times5=15$ 个 Token 槽位；Packed/Ragged 布局只存放 $3+5+2=10$ 个有效 Token：

```text
[A1 A2 A3 B1 B2 B3 B4 B5 C1 C2]
```

若上面的符号表示 Token ID，底层存储可以写成：

```text
token_ids.shape = [10]
cu_seqlens      = [0, 3, 8, 10]
```

经过 Embedding 后，对应的隐藏状态形状才是 `hidden_states.shape = [10, hidden_size]`。

其中边界数组表示：

```text
sequence 0 = tokens[0:3]
sequence 1 = tokens[3:8]
sequence 2 = tokens[8:10]
```

`Ragged` 描述逻辑上不规则的序列形状，`Packed` 描述物理上把有效 Token 连续存放。真正能否完全跳过 Padding 计算，还取决于 Attention、位置编码和其他算子是否支持变长布局；不能只改变张量形状而继续调用要求规则矩阵的内核。

## 请求何时进入和退出？

### Static Batching 的 Batch Drain

Packed/Ragged Batching 解决的是一次 Forward 内不同长度请求如何高效存放，Continuous Batching 解决的则是不同生命周期的请求何时进入和退出 Batch。

假设 4 个请求的输出长度分别为：

```text
A: 10 tokens
B: 20 tokens
C: 100 tokens
D: 1000 tokens
```

传统 Static Batching 会先组成：

```text
Batch = [A B C D]
```

Decode 到第 10 步时，A 已经完成，但 B、C、D 仍在生成；如果 A 的位置不能立即交给新请求，实际 Batch Size 就会从 4 逐渐降到 3，最后只剩 1。这种现象常被称为 Batch Drain，会造成 GPU 利用率下降。

### Continuous Batching：按迭代动态补位

Continuous Batching 可以用下面的概念模型理解：

```text
初始： [A B C D]
第 10 步后：A 完成，E 进入 → [E B C D]
第 20 步后：B 完成，F 进入 → [E F C D]
```

调度器不再等整个 Batch 完成，而是以一次模型迭代为调度单位。每轮 Decode 后，它都会检查：

1. 哪些请求已经结束；
2. 哪些请求还要继续生成；
3. 释放的资源能否接纳等待队列中的新请求。

真实系统并不是简单替换固定槽位：新请求 E 往往还要先执行 Prefill，能否与 B、C、D 的 Decode 同轮运行，还受到 Token Budget、KV Cache 容量、优先级和 Chunked Prefill 策略约束。核心在于调度器每个 iteration 都会重建执行计划，从而让运行中的请求数长期保持在较高水平。2022 年 OSDI 论文 [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) 系统化提出了迭代级调度，并成为 Continuous Batching 的重要基础。

### Chunked Prefill：让长 Prompt 分轮进入

有了 Token Budget 之后，还会遇到一个新问题：如果本轮预算是 8,192 Token，而某个 Prompt 长达 50,000 Token，它无法一次装入预算。

Chunked Prefill 会把这段 Prefill 拆成多个可调度单元。每轮只处理其中一部分，并根据当轮剩余的 Token Budget 安排实际 Chunk 大小。因此，50,000 Token 不一定机械地切成固定的 8,192 Token；若同一轮还需要优先调度 Decode，请求能使用的 Prefill 预算会更小。

vLLM 的调度配置也明确说明：启用 Chunked Prefill 后，Prefill 会依据剩余的 `max_num_batched_tokens` 被分块。

Chunked Prefill 让超长 Prompt 不必独占一次巨大的 Forward，也不会因为超过单轮预算而一直停留在等待队列中；同时，它允许 Decode 与 Prefill 在同一调度体系下共享 Token Budget。代价是调度和中间状态管理更复杂，而且预算配置会影响 TTFT、ITL/TPOT 与吞吐之间的平衡。

## 一次请求如何完成 Prefill 和 Decode？

把前面的技术串起来，一批请求会经历以下过程：

1. **进入等待队列**：请求携带不同的输入长度和预期输出长度进入系统。
2. **选择本轮请求**：Length Bucketing 可以优先组合长度相近的请求；Token Budget 与最大请求数共同限制本轮工作量。
3. **执行 Prefill**：规则矩阵路径可使用 Dynamic Padding；支持变长内核时可使用 Packed/Ragged 布局，只处理有效 Token。
4. **拆分超长 Prefill**：如果输入无法一次装入预算，Chunked Prefill 把它拆成多轮可调度工作。
5. **进入 Decode**：请求每轮生成一个或少量 Token，调度器持续更新运行状态。
6. **完成后立即补位**：Continuous Batching 移除已完成请求，并从等待队列补入新请求。

这条链路说明，长度感知调度并不是单一算法，而是一个跨越准入、数据布局和生命周期管理的系统问题。

## vLLM 与 SGLang 中的对应参数

下面以 vLLM v0.31.0 和 SGLang v0.5.21 为例，列出原文所讨论技术在两个主流推理框架中的常用入口。内容核验于 2026 年 10 月 8 日；参数语义和默认值可能随版本变化，部署前应以目标版本文档为准。

| 框架 | 参数 | 作用 |
| --- | --- | --- |
| vLLM | `--max-num-batched-tokens` | 限制单次迭代调度的 Token 数 |
| vLLM | `--max-num-seqs` | 限制单次迭代中的序列数 |
| vLLM | `--enable-chunked-prefill` / `--no-enable-chunked-prefill` | 控制是否允许分块 Prefill；V1 在受支持路径下会尽可能启用 |
| SGLang | `--max-running-requests` | 限制同时运行的请求数 |
| SGLang | `--chunked-prefill-size` | 控制分块 Prefill 的大小或调度预算 |
| SGLang | `--max-prefill-tokens` | 控制 Prefill 批次的 Token 预算与准入 |

vLLM 可参考 [Optimization and Tuning](https://docs.vllm.ai/en/v0.31.0/configuration/optimization/) 与 [SchedulerConfig](https://docs.vllm.ai/en/v0.31.0/api/vllm/config/scheduler/)；SGLang 可参考其 [Hyperparameter Tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning) 文档。这里使用版本化链接核对 vLLM 行为，是为了避免把某个版本的默认值误当成长期不变的结论。

## 调优时应该观察什么？

这类参数不能只看峰值吞吐。至少应同时观察：

- **TTFT**：请求从到达到生成首个 Token 的时间；
- **ITL/TPOT**：Decode 阶段相邻输出 Token 的间隔；
- **吞吐量**：单位时间处理的请求数或 Token 数；
- **排队时间**：请求为了等待同类长度或空闲预算而停留多久；
- **Padding 比例**：规则矩阵路径中无效 Token 槽位占比；
- **每轮调度 Token 数**：是否长期低于预算，或因预算过高造成延迟抖动；
- **GPU 利用率与显存占用**：计算是否吃满，以及 KV Cache 是否成为约束。

例如，提高 Token Budget 往往能让短 Prefill 组成更大的批次，但也可能延长单轮执行时间，进而影响正在运行的 Decode 请求。合理的配置来自目标流量分布和服务等级目标，而不是单纯把参数调到最大。

## 边界与常见误区

- **Length Bucketing 不等于完整的长度感知调度。** 它只决定哪些请求更适合放在一起，无法独自解决单轮预算、数据布局和动态补位问题。
- **Dynamic Padding 不是无 Padding。** 它仍会补齐序列，只是补到当前批次最大长度。
- **Packed/Ragged Batching 不等于 Continuous Batching。** 前者是一次执行中的数据布局，后者是跨迭代的请求生命周期调度。
- **Token Budget 越大不一定越好。** 更大的预算可能提升吞吐，也可能拉长迭代时间、加重显存压力并影响 Decode 延迟。
- **Chunked Prefill 不是固定长度切块。** 实际 Chunk 会受到本轮剩余预算和其他运行请求的影响。
- **分桶越细不一定越高效。** 过度等待长度完全匹配的请求，可能增加队列延迟并造成流量碎片。

## 总结

从 Padding 到 Continuous Batching，优化目标始终是让 GPU 把时间和显存花在有效 Token 上：

- Length Bucketing 与 Dynamic Padding 减少同一批次中的长度差异和补齐范围；
- Token-aware Batching 用 Token 数而不是固定请求数描述单轮负载；
- Packed/Ragged Batching 在数据布局上只保存有效 Token；
- Continuous Batching 让已完成请求及时退出、新请求及时补位；
- Chunked Prefill 把超长 Prompt 拆成符合单轮预算的工作单元。

真正有效的推理优化，需要把这些机制放在同一个调度闭环里，并用 TTFT、ITL/TPOT、吞吐、排队时间、显存和 GPU 利用率共同验证。
