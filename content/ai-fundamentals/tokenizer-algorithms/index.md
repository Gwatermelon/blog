---
title: "Tokenizer 基础：从文本到 Token ID"
date: 2026-07-15
lastmod: 2026-07-15
draft: false
description: "系统梳理 Tokenizer 的处理流水线，以及词级、字符级、BPE、WordPiece、Unigram 和字节级 BPE 的原理、取舍与常见误区。"
summary: "Tokenizer 不只是切词：它要在词表大小、序列长度、开放词汇与多语言覆盖之间权衡。本文用例子和一份可运行的 BPE 实现讲清主流算法。"
tags: ["AI 基础", "LLM", "Tokenizer", "BPE", "WordPiece", "Unigram"]
categories: ["AI 基础原理"]
ShowToc: true
TocOpen: true
---

> **核心直觉：** Tokenizer 是文本与神经网络之间的编码层。它把字符串转换成有限词表中的整数 ID，同时尽量让词表不要太大、序列不要太长，并让任意输入都能被表示。

## Tokenizer 到底做什么

语言模型不能直接计算字符串。模型接收的是整数 ID，再通过 embedding 表把每个 ID 映射成向量。以文本 `playing football` 为例，一种可能的结果是：

```text
原始文本：playing football
Token：   ["play", "ing", " football"]
Token ID：[1234, 567, 8910]
```

这里的切分只用于说明流程；真实结果由具体模型的 tokenizer、词表和配置决定。同一段文本交给不同模型，token 数量和 ID 通常都不相同。

一个完整的 tokenizer 通常包含以下阶段：

1. **归一化（Normalizer）**：按配置处理 Unicode、大小写或重音符号。归一化不是越多越好；例如代码模型往往必须保留大小写和空白差异。
2. **预切分（Pre-tokenizer）**：按空格、标点或正则规则划分边界，并记录原文位置。SentencePiece 一类方案也可以直接从原始句子训练，不依赖语言专用的分词器。
3. **子词模型（Model）**：使用 BPE、WordPiece 或 Unigram 等算法把片段切成词表中的 token。
4. **后处理（Post-processor）**：按模型约定加入 `[CLS]`、`[SEP]`、BOS、EOS 等特殊 token。
5. **ID 映射**：根据固定词表把 token 转换成整数 ID。

编码只是半个过程。Tokenizer 还需要把模型生成的 ID 解码回文本，并尽可能正确地恢复空格、标点和字节序列。

## 设计时的五个权衡

Tokenizer 的目标不是切出语言学上最正确的词，而是为模型构造合适的离散表示。常见权衡包括：

- **词表大小**：词表越大，输入 embedding 和输出投影层通常越大。
- **序列长度**：切分越细，token 数越多，训练与推理成本越高，可容纳的有效文本也越少。
- **开放词汇**：新词、人名、URL、代码、emoji 和拼写错误不应轻易退化成 `[UNK]`。
- **跨语言覆盖**：词表容量要在不同语言、字符和领域之间合理分配。
- **可逆性与稳定性**：解码应尽量还原原文；同一配置下的编码结果应可复现。

这些目标彼此冲突。词级切分序列短，却需要巨大词表；字符级词表小，却会显著拉长序列。子词方法位于两者之间，因此成为现代 Transformer 的主流选择。

## 从词级到子词级

### 词级 Tokenizer

词级方法把完整单词作为 token：

```text
I love playing football
→ ["I", "love", "playing", "football"]
```

它直观且序列较短，但自然语言的词形变化、复合词和专有名词几乎没有上限。只保留高频词会产生大量 `[UNK]`，全部保留又会让词表迅速膨胀。因此，纯词级 tokenizer 在现代大模型中已经少见。

### 字符级 Tokenizer

字符级方法把每个 Unicode 字符作为 token：

```text
playing → ["p", "l", "a", "y", "i", "n", "g"]
我喜欢你 → ["我", "喜", "欢", "你"]
```

它能用较小的词表覆盖大量文本，但会生成很长的序列，而且 Unicode 字符集合本身仍然很大。纯字符建模可用于特定的 OCR、拼写纠错等任务，却不是当前通用大模型最常见的方案。

### 子词 Tokenizer

子词方法保留高频片段，把低频词拆成更小单位：

```text
playing      → ["play", "ing"]
unbelievable → ["un", "believ", "able"]
```

这样既能复用词根、前缀和后缀，又能控制词表大小。需要注意的是，算法只根据训练目标和语料统计学习片段，并不保证 token 边界符合语言学词素。

## BPE：从小单位不断合并

BPE（Byte Pair Encoding）最初是一种数据压缩算法，后来被改造成子词学习方法。用于文本时，它通常从字符或字节等小单位开始，反复合并语料中最频繁的相邻 pair。

假设词频为：

```text
low     × 5
lower   × 2
lowest  × 2
```

初始时，每个词被拆成字符并带上词尾标记：

```text
l o w </w>
l o w e r </w>
l o w e s t </w>
```

训练循环如下：

1. 按词频加权，统计所有相邻 token pair；
2. 选择出现次数最多的 pair；
3. 把它合并为新 token，并记录合并规则；
4. 更新语料的切分；
5. 重复执行，直到达到词表大小或合并次数上限。

如果 `(l, o)` 最常见，就先得到 `lo`；后续可能继续把 `(lo, w)` 合并成 `low`。最终编码新文本时，按照训练得到的合并优先级应用规则。

### 一份可运行的简化实现

下面的代码保留了 BPE 的核心机制：词频加权、pair 统计、确定性 tie-break、合并规则训练，以及用规则编码新词。为了突出算法，它没有实现 Unicode 归一化、特殊 token、字节回退或持久化格式，不能直接替代生产级 tokenizer。

```python
from __future__ import annotations

from collections import Counter
import re


Word = tuple[str, ...]
Pair = tuple[str, str]


def pre_tokenize(text: str) -> list[str]:
    """保留单词与标点；这里只做教学用的简单预切分。"""
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def merge_pair(symbols: Word, pair: Pair) -> Word:
    """合并序列中所有不重叠的指定 pair。"""
    merged: list[str] = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i : i + 2] == pair:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


def count_pairs(vocabulary: Counter[Word]) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    for symbols, frequency in vocabulary.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += frequency
    return counts


def train_bpe(corpus: list[str], num_merges: int) -> list[Pair]:
    word_counts = Counter(
        token
        for text in corpus
        for token in pre_tokenize(text)
    )
    vocabulary: Counter[Word] = Counter({
        tuple(word) + ("</w>",): frequency
        for word, frequency in word_counts.items()
    })

    merges: list[Pair] = []
    for _ in range(num_merges):
        pair_counts = count_pairs(vocabulary)
        if not pair_counts:
            break

        # 频率相同时按字典序选择，保证示例结果可复现。
        best_pair = min(
            pair_counts,
            key=lambda pair: (-pair_counts[pair], pair),
        )
        merges.append(best_pair)

        updated: Counter[Word] = Counter()
        for symbols, frequency in vocabulary.items():
            updated[merge_pair(symbols, best_pair)] += frequency
        vocabulary = updated

    return merges


def encode_word(word: str, merges: list[Pair]) -> list[str]:
    symbols: Word = tuple(word.lower()) + ("</w>",)
    merge_rank = {pair: rank for rank, pair in enumerate(merges)}

    while len(symbols) > 1:
        candidates = {
            pair for pair in zip(symbols, symbols[1:])
            if pair in merge_rank
        }
        if not candidates:
            break
        best_pair = min(candidates, key=merge_rank.__getitem__)
        symbols = merge_pair(symbols, best_pair)

    # 词尾标记只用于防止跨越词边界的合并，不作为最终文本输出。
    return [
        symbol.removesuffix("</w>")
        for symbol in symbols
        if symbol != "</w>"
    ]


corpus = [
    "low low low low low",
    "lower lower",
    "lowest lowest",
]
rules = train_bpe(corpus, num_merges=8)

print(rules)
print(encode_word("lowest", rules))
```

真实实现还需要处理预切分边界、未知字符、词表 ID、特殊 token、归一化和高效数据结构。不同实现对词尾、空格和合并冲突的约定也可能不同。

### BPE 的优缺点

BPE 简单、训练与编码都较高效，而且合并规则固定后结果具有确定性。它的局限是训练目标主要来自相邻片段频率，并不直接优化语言学边界或下游模型损失；稀有字符仍可能占用词表或触发未知字符问题，具体取决于基础符号集合和回退策略。

## WordPiece：训练打分与最长匹配

WordPiece 与 BPE 都会学习子词词表，但不能简单理解为“BPE 换了一个 token 前缀”。常见 WordPiece 训练实现不会只选择频率最高的 pair，而会使用类似下面的分数，降低两个高频片段仅因各自常见而被合并的倾向：

```text
score(a, b) = freq(a, b) / (freq(a) × freq(b))
```

推理时，WordPiece 通常对每个预切分后的词执行**最长匹配优先**：先尝试词表中最长的前缀；匹配后，再对剩余部分重复。如果无法把整个词拆成已知片段，经典实现可能输出 `[UNK]`。

```text
playing → ["play", "##ing"]
```

`##` 是一种显示与解码约定，表示该 token 接在词内而不是从词首开始；它不是 WordPiece 的数学本质。

## Unigram：从大词表逐步剪枝

Unigram 与 BPE 的生长方向相反：它先构造一个较大的候选子词集合，再逐步删除对语料似然贡献较小的 token。

在简化的 Unigram 模型中，一种切分 (x = (x_1, \ldots, x_n)) 的概率为：

\[
P(x) = \prod_{i=1}^{n} P(x_i)
\]

同一个字符串可能存在多条切分路径：

```text
unhappiness
→ ["un", "happiness"]
→ ["un", "happy", "ness"]
→ ["u", "n", "h", ...]
```

编码时可以用动态规划寻找概率最大的路径。训练时则反复估计 token 概率、评估删除候选造成的损失，再剪掉影响较小的候选，直到达到目标词表规模。

因为模型保留了切分概率，训练阶段还可以从多种合理切分中采样，这就是 **subword regularization** 的基础。它能把分词歧义作为噪声注入训练，但实现和训练成本通常高于确定性的 BPE。

## 字节级 BPE：用 256 个字节兜底

字符级 BPE 的基础词表取决于 Unicode 字符集合。面对中文、日文、emoji、代码和噪声文本，大量低频字符会占据词表空间；训练时未覆盖的字符还可能变成未知 token。

字节级 BPE（Byte-level BPE / BBPE）先把 UTF-8 文本转换成字节序列，基础字节只有 256 种，再在字节或已合并的字节片段上执行 BPE。只要输入能编码为字节，就不必因为新字符而使用 `[UNK]`。

以 ASCII 文本为例：

```text
cat → [99, 97, 116]
car → [99, 97, 114]
```

如果 `(99, 97)` 最常见，可以先合并代表 `ca` 的片段，之后再继续合并 `cat`。对非 ASCII 字符，一个人类眼中的字符可能由多个 UTF-8 字节组成，因此中间 token 不一定对应可单独阅读的字符。

GPT-2 风格实现会把 256 个字节可逆地映射到可显示的 Unicode 符号，再运行 BPE。调试输出中常见的 `Ġ` 通常用来表示带有前导空格的片段；它是字节到可见符号映射的一部分，不是原文真的包含了这个字母。

字节级方法消除了字符层面的 OOV，并促进跨语言共享基础词表，但不保证所有语言的压缩效率相同。某些字符需要多个 UTF-8 字节，低资源语言或特殊领域仍可能被切得更碎。

## SentencePiece 不是第四种子词算法

SentencePiece 经常与 BPE、WordPiece、Unigram 并列，容易造成概念混淆。更准确地说，SentencePiece 是一个从原始句子训练、编码和解码子词模型的工具体系；它支持 BPE 和 Unigram 等模型，并用特殊符号显式表达空格，从而实现不依赖语言专用预分词器的处理流程。

因此，“使用 SentencePiece”并不能单独说明底层采用了哪种子词算法，还需要查看模型类型和具体配置。

## 主流方法对比

| 方法 | 起始单位 | 训练方向或核心规则 | 推理时切分 | 主要特点 |
| --- | --- | --- | --- | --- |
| 词级 | 完整单词 | 统计并截断词表 | 查词表 | 序列短，但词表大、OOV 严重 |
| 字符级 | Unicode 字符 | 通常无需合并 | 逐字符 | 词表较小，但序列长 |
| BPE | 字符或字节 | 反复合并高频相邻 pair | 按合并优先级应用规则 | 简单、高效、确定性强 |
| WordPiece | 字符/子词 | 常用归一化 pair 分数扩展词表 | 最长匹配优先 | BERT 系模型中的经典方案，可能产生 `[UNK]` |
| Unigram | 大量候选子词 | 概率估计并逐步剪枝 | 选择最高概率路径，也可采样 | 支持多种切分，训练更复杂 |
| 字节级 BPE | 256 种字节 | 在字节片段上执行 BPE | 按 merge rules 合并 | 字符层面无 OOV，适合多语言和噪声文本 |

## Tokenizer 如何影响模型

Tokenizer 并不是无关紧要的预处理工具，它会改变模型看到的数据形态：

- **上下文容量**：上下文窗口按 token 计数；切分更碎时，同样长度的原文会更快占满窗口。
- **训练和推理成本**：token 数量影响前向次数、KV Cache 占用和计费；词表大小又影响 embedding 与词表投影成本。
- **跨语言效率**：不同语言的每 token 字符数可能差异明显，进而影响可用上下文和使用成本。
- **数字、代码与结构化文本能力**：数字、缩进、运算符和标识符如何切分，会改变模型学习局部模式的难度。
- **模型兼容性**：tokenizer 的词表、ID 排列、特殊 token 和归一化规则都属于模型参数契约，不能随意替换。

评估 tokenizer 时，不应只看词表大小。至少要在目标语种和领域上比较 token/字符比、字节/token 比、未知 token 率、解码可逆性，以及加入特殊 token 后的真实序列长度。

## 常见误区

### Token 就等于单词

Token 可能是完整单词、子词、标点、空白片段、单个字节或特殊控制符。界面里显示的“token 数”不能直接等同于词数。

### 子词边界一定符合词根和词缀

BPE 和常见 WordPiece 训练主要依赖语料统计。`play + ing` 可能恰好符合词法直觉，但算法并不承诺每次都得到语义上最自然的边界。

### Byte-level BPE 的 token 就是一个字节

它以字节为基础词表，但 BPE 会把多个相邻字节反复合并。最终 token 可以覆盖一个字节、一个字符、多个字符，甚至带前导空格的整段文本。

### 词表越大越好

更大的词表可能缩短序列，却会增加参数和词表投影开销，还会让低频 token 得不到充分训练。最优规模取决于语料、模型大小、语言分布和部署目标。

### 可以给训练好的模型随意更换 Tokenizer

模型 embedding 的第 `i` 行只对应训练时词表中的第 `i` 个 token。更换词表或 ID 顺序会让这些语义对应关系失效；即使词表大小相同，也不能直接替换。

## 总结

Tokenizer 的本质是在**词表容量**与**序列长度**之间做压缩，并保证开放世界中的文本能够进入一个固定大小的模型接口。

- 词级方法直观但无法经济地覆盖开放词汇；
- 字符级方法覆盖广，却会产生过长序列；
- BPE 通过高频 pair 合并构造确定性的子词词表；
- WordPiece 常用不同的训练打分，并在推理时执行最长匹配；
- Unigram 用概率模型比较多条切分路径，并从大候选集逐步剪枝；
- 字节级 BPE 用 256 个基础字节解决字符 OOV，再通过合并恢复压缩效率。

理解这些差异后，再看到模型的词表大小、上下文窗口或 token 计费时，就能把它们与底层文本表示联系起来，而不只是把 tokenizer 当成一个黑盒 `encode()` 函数。

## 参考资料

- [Hugging Face：Tokenization algorithms](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Hugging Face Tokenizers：Components](https://huggingface.co/docs/tokenizers/components)
- [Sennrich et al.：Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [Kudo：Subword Regularization](https://arxiv.org/abs/1804.10959)
- [Kudo & Richardson：SentencePiece](https://aclanthology.org/D18-2012/)
- [Wang et al.：Neural Machine Translation with Byte-Level Subwords](https://arxiv.org/abs/1909.03341)
