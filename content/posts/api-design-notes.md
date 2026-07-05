---
title: "设计一个让调用者省心的 HTTP API"
date: 2026-06-28T10:00:00+08:00
description: "从资源建模、错误语义到幂等性，梳理 API 设计中最值得提前想清楚的事。"
summary: "API 的质量不只体现在能否返回数据，更体现在调用者能否准确预测它的行为。"
categories: ["后端开发"]
tags: ["API", "HTTP", "架构"]
series: ["工程基本功"]
---

一个好 API 的核心不是“RESTful 程度”，而是**可预测性**。调用者看到路径、方法和响应，就能推断它会做什么、失败时怎样恢复、重复调用是否安全。

## 先设计资源，再设计路径

把领域中的名词找出来，再确定它们之间的关系：

```http
GET    /projects/{projectId}/tasks
POST   /projects/{projectId}/tasks
PATCH  /tasks/{taskId}
DELETE /tasks/{taskId}
```

路径表达资源，方法表达动作。比起 `/getTaskList`，这种结构更容易形成一致的权限、日志与缓存策略。

## 错误响应要能指导行动

错误码服务于程序判断，错误信息服务于人类排查。一个稳定的错误结构可以是：

```json
{
  "code": "TASK_STATE_CONFLICT",
  "message": "当前状态不允许完成任务",
  "requestId": "req_01J...",
  "details": { "currentState": "archived" }
}
```

不要让客户端解析自然语言来判断错误类型，也不要把内部堆栈暴露到公网响应中。

## 对重试保持敬畏

网络超时只表示客户端没有收到结果，并不表示服务端没有执行。支付、创建订单等操作应支持幂等键，让相同业务意图的重试返回同一个结果。

API 一旦发布就是承诺。命名可以朴素，行为必须稳定；文档可以逐步丰富，错误语义和兼容策略最好从第一天就认真对待。

