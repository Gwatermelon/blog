---
title: "Research Todo"
description: "记录当前真正需要推进的算法题、数学知识、推理加速问题和算子实现任务。"
layout: "todo"
ShowToc: false
taskGroups:
  - title: "LeetCode 算法题"
    subtitle: "完成解题、复杂度分析与过程复盘"
    tasks:
      - id: "leetcode-binary-tree-max-path-sum"
        text: "二叉树的最大路径和如何写"
        note: "整理递归状态、全局最优值与边界情况"
      - id: "leetcode-longest-subarray"
        text: "最长 Subarray 如何写"
        note: "明确题目约束后整理滑动窗口或动态规划方法"
  - title: "大模型推理加速"
    subtitle: "围绕投机推理与缓存机制回答关键问题"
    tasks:
      - id: "inference-qkv-kv-cache"
        text: "QKV 为什么只有 KV Cache"
        note: "从自回归注意力中的复用关系讲清楚原因"
  - title: "数学知识"
    subtitle: "整理数学概念的直觉、推导与应用"
    tasks:
      - id: "math-taylor-expansion"
        text: "泰勒展开"
        note: "梳理展开公式、余项、收敛条件与常见函数示例"
  - title: "算子实现与优化"
    subtitle: "从数学定义走到面向硬件的高效实现"
    tasks:
      - id: "kernel-softmax"
        text: "Softmax 如何实现和优化"
        note: "覆盖数值稳定性、并行归约、访存与融合"
      - id: "kernel-flash-attention"
        text: "讲解 FlashAttention"
        note: "讲清 IO-aware 思路、分块计算与在线 Softmax"
---

这 6 项是当前预置任务。你也可以在页面底部选择分类并添加新的研究事项；自定义事项和勾选状态均保存在当前浏览器中。
