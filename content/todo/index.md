---
title: "Research Todo"
description: "记录近期工作安排、技术调研和论文阅读计划。"
layout: "todo"
ShowToc: false
taskGroups:
  - title: "今日安排"
    subtitle: "收敛到今天真正需要推进的三件事"
    tasks:
      - id: "today-spec-model"
        text: "整理投机解码的基础性能模型"
        note: "明确草稿成本、接受率与验证成本三个变量"
      - id: "today-metrics"
        text: "确定第一版实验指标"
        note: "TTFT、TPOT、吞吐量、平均接受长度、显存"
      - id: "today-paper"
        text: "精读 Speculative Decoding 算法部分"
        note: "独立推导接受与修正分布"
  - title: "本周调研"
    subtitle: "围绕核心研究问题建立可验证的知识地图"
    tasks:
      - id: "week-baseline"
        text: "搭建自回归解码性能基线"
        note: "固定模型、硬件、batch size 和采样参数"
      - id: "week-draft"
        text: "调研草稿模型选择策略"
        note: "模型规模、分布接近程度、延迟和接受率"
      - id: "week-framework"
        text: "对比主流推理框架的投机解码支持"
        note: "重点记录接口、限制与可观测指标"
      - id: "week-roofline"
        text: "补齐 Roofline Model 与算术强度"
        note: "解释不同 batch size 下的瓶颈迁移"
  - title: "长期选题池"
    subtitle: "暂不承诺时间，但值得持续追踪"
    tasks:
      - id: "backlog-self-spec"
        text: "自投机解码与 early-exit 路线"
        note: "减少独立草稿模型带来的显存和调度成本"
      - id: "backlog-mtp"
        text: "Multi-Token Prediction 与推理加速"
        note: "训练目标如何影响部署阶段的候选质量"
      - id: "backlog-quant"
        text: "量化对投机接受率的影响"
        note: "草稿与目标模型量化误差是否会放大分布差异"
      - id: "backlog-hardware"
        text: "不同 GPU 架构上的收益迁移"
        note: "带宽、算力与 kernel launch 开销的共同作用"
---

勾选状态和临时笔记保存在当前浏览器中；长期计划仍由本页 Markdown 维护，避免研究记录被锁在某台设备里。

