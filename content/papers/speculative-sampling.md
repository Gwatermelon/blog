---
title: "论文速读：Accelerating Large Language Model Decoding with Speculative Sampling"
date: 2026-07-04T15:00:00+08:00
description: "从采样等价性出发理解投机采样，以及它在不同任务与模型组合下的收益边界。"
summary: "与投机解码同期的重要工作，重点关注无损采样、草稿模型选择和真实推理收益。"
categories: ["大模型推理加速"]
tags: ["投机采样", "论文研读", "LLM Inference"]
series: ["投机推理"]
paperStatus: "待读"
paperVenue: "arXiv 2023"
paperDomain: "推理加速"
paperUrl: "https://arxiv.org/abs/2302.01318"
---

## 阅读目标

这篇论文将作为投机推理基础阅读的第二篇，重点与 *Fast Inference from Transformers via Speculative Decoding* 对照：

- 两篇工作的算法表述与证明路径有何差异；
- 接受/拒绝过程如何保证采样结果不发生偏移；
- 不同草稿模型、任务和温度设置如何影响收益；
- 论文中的理想加速与当前推理框架实现之间存在哪些落差。

## 待读清单

- [ ] 画出完整采样流程；
- [ ] 独立推导修正分布；
- [ ] 整理论文全部评测指标；
- [ ] 与基础投机解码实现进行逐项对照。

原文：[arXiv:2302.01318](https://arxiv.org/abs/2302.01318)

