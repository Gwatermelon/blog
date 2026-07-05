---
title: "论文速读：Medusa — Multiple Decoding Heads"
date: 2026-07-03T09:00:00+08:00
description: "用多个解码头并行预测未来 token，探索不依赖独立草稿模型的推理加速路线。"
summary: "把 Medusa 放入自投机与多 token 预测脉络中，重点分析额外解码头、树状注意力和系统收益。"
categories: ["大模型推理加速"]
tags: ["Medusa", "多Token预测", "论文研读"]
series: ["投机推理"]
paperStatus: "待读"
paperVenue: "ICML 2024"
paperDomain: "推理加速"
paperUrl: "https://arxiv.org/abs/2401.10774"
---

## 阅读目标

Medusa 不再依赖一套独立的小型草稿模型，而是在目标模型上增加多个解码头来预测未来 token。计划重点回答：

- 多个解码头分别学习什么目标？
- 候选树如何构造并在一次前向中验证？
- 训练额外解码头的成本与适用边界是什么？
- 与双模型投机解码相比，它减少了什么开销，又增加了什么约束？

## 待读清单

- [ ] 梳理 Medusa-1 与 Medusa-2 的差异；
- [ ] 理解 tree attention 的掩码构造；
- [ ] 记录模型质量、延迟和吞吐量指标；
- [ ] 调研主流推理框架的实现支持。

原文：[arXiv:2401.10774](https://arxiv.org/abs/2401.10774)

