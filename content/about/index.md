---
title: "关于我"
layout: "single"
ShowToc: true
---

我是张革，目前从事大模型推理加速、AI Agent、程序分析与自动算子优化相关工作。近期重点关注 Qwen 系列模型的投机推理加速，以及基于 MCTS 的自动算子优化，覆盖算法设计、模型训练、推理框架适配、自动化评测和端到端性能优化。

联系方式：

- Email: 772422673@qq.com
- GitHub: [Gwatermelon](https://github.com/Gwatermelon)

## 工作经历

### 北京华为数字技术有限公司

2022 年 1 月至今

- 高级工程师 B / Committer，2026 年至今
- 工程师 A / Committer，2024 年至 2026 年
- 工程师 B，2022 年至 2024 年

### 滴滴出行

2021 年 7 月至 2021 年 10 月，实习

## 项目经历

### 大模型推理加速

2026 年至今，Team Leader

- 负责 Qwen3 / Qwen3.5 系列大模型投机推理加速，基于 vLLM Speculators 完成 EAGLE-3 草稿模型训练、推理适配与端到端验证。
- 在中英混合数据集上实现接受步长 3.5 的性能收益，高于开源 EAGLE-3 模型效果。
- 基于 FrSpec 设计并实现 Qwen3.5 MTP 一键式推理优化流程，通过调整 vLLM 中草稿模型的词表范围降低草稿阶段推理耗时。
- 该流程在单请求场景提升投机解码端到端性能收益 10%，在 16 并发请求下仍获得 5% 的性能收益。
- 面向 Qwen3.5 系列训练 DFlash 草稿模型，建立草稿质量、接受率、吞吐、延迟等评测流程，在内部 benchmark 上取得当前最优加速收益。

### 算子优化与 Agent 开发

2025 年至 2026 年，Core Developer

- 对标开源技术 EVO，设计并实现基于蒙特卡洛树搜索（MCTS）的 LLM 算子优化 Agent，将人工算子调优流程转化为多 Agent 自主搜索与迭代优化流程。
- 负责 MCTS 搜索策略、多 Agent 任务编排与调度逻辑开发，支持复杂算子优化任务的自动拆解、候选代码生成、性能评测与反馈迭代。
- 在长序列及复杂算子场景下提升候选生成质量与搜索效率，实现核心算子倍数级性能提升。

### 鸿蒙生态程序分析底座构建

2024 年至 2025 年，Team Leader

- 基于鸿蒙生态程序分析底座方舟分析器 ArkAnalyzer 引入 C/C++ 语言解析能力，实现 TS 与 C/C++ 在 ArkAnalyzer 中的统一 IR 翻译与表达。
- 主导端到端架构设计、核心技术攻关与可行性验证，组织团队完成 C/C++ 核心语法及常用标准库的分析能力交付。
- 将 C++ 特性深度接入 ArkIR，实现万行级代码语义秒级解析。
- 基于该底座开发的检查工具在关键版本发布前拦截数十项高危内存管理及性能规范违例问题。
- 项目地址：[openharmony-sig/arkanalyzer](https://gitcode.com/openharmony-sig/arkanalyzer/tree/ArkAnalyzer-cpp-tooling)

### 相似算子识别

2023 年至 2024 年，Core Developer

- 面向 HPC 算子替换场景，设计自动化相似算子识别方案，为高性能算子替换提供候选代码块匹配能力。
- 基于 Joern 引入 Fortran 语言语义表达，构建面向 Fortran / C / C++ 算子的代码属性图表示。
- 基于代码属性图实现类 Bag of Paths 的路径匹配算法，在超大规模代码库中实现高准确率算子相似度匹配。
- 方案集成至核心产品线算子优化系统，提升算子替换工程的自动化率。

## 论文与专利

- [TechReport 2024] Wenrui Zhang, Tiehang Fu, Ting Yuan, Ge Zhang, Dong Chen, and Jie Wang. "A Lightweight Framework for Adaptive Retrieval In Code Completion With Critique Model." arXiv:2406.10263.
- [Patent 2026] 一种融合大模型生成与蒙特卡洛树搜索策略的算子性能迭代优化系统。
- [Patent 2025] 一种领域特定语言语句生成方法、装置以及计算设备集群。

## 教育经历

- 爱丁堡大学，Master of Science in Computer Science，高性能计算与数据科学，2020 年 8 月至 2021 年 12 月。
- 中国计量大学，Bachelor of Science，信息与计算科学，2016 年 8 月至 2020 年 7 月。

## 技术栈

- 大模型推理加速：Speculative Decoding、MTP、PEARL、SSD、EAGLE-3、DFlash、vLLM、vLLM Speculators。
- AI Agent 与算子优化：MCTS、多 Agent 编排、任务规划、候选代码生成、性能反馈迭代。
- 代码智能与程序分析：Clang AST、tree-sitter、Joern、代码属性图、ArkAnalyzer、ArkIR。
- 工程与语言：Python、Git、Docker、Linux。
