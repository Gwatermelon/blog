# 博客文章添加指南（无需调用大模型）

这份指南面向不熟悉 Hugo、HTML 或 Go 模板的内容作者。日常添加文章时，你只需要处理 **文件夹、Markdown 文件和图片**，不需要修改网站程序。

## 1. 先记住三个规则

1. 每篇文章使用一个独立文件夹，正文文件固定命名为 `index.md`。
2. 文件夹名称使用英文小写和连字符，例如 `python-memory-model`，不要使用空格和中文。
3. 不要手动修改 `public/`、`layouts/`、`themes/` 和 `assets/`；这些属于网站程序或构建产物。

## 2. 选择文章所属模块

| 想写的内容 | 放入目录 | 复制的模板 |
| --- | --- | --- |
| 普通技术文章、工作总结 | `content/posts/` | `docs/templates/posts.md` |
| 数学、机器学习、Transformer 基础 | `content/ai-fundamentals/` | `docs/templates/ai-fundamentals.md` |
| 大模型推理、KV Cache、量化、服务优化 | `content/model-inference/` | `docs/templates/model-inference.md` |
| 论文阅读与复现笔记 | `content/papers/` | `docs/templates/papers.md` |
| LeetCode 解题记录 | `content/leetcode/` | `docs/templates/leetcode.md` |

无论文章放在哪个模块，发布后都会自动出现在「文章归档」中，并按日期从新到旧排列。

## 3. 最简单的添加流程

以下示例要添加一篇 AI 基础文章，文件夹名为 `transformer-attention`。

在项目根目录打开 PowerShell，依次运行：

```powershell
New-Item -ItemType Directory -Path "content\ai-fundamentals\transformer-attention"
Copy-Item "docs\templates\ai-fundamentals.md" "content\ai-fundamentals\transformer-attention\index.md"
```

然后打开：

```text
content/ai-fundamentals/transformer-attention/index.md
```

完成下面三件事即可：

1. 把模板中的标题、日期、简介和标签替换为真实内容；
2. 删除正文中的提示文字，写入自己的文章；
3. 如果有图片，把图片放进同一个文件夹。

最终目录类似：

```text
content/
└── ai-fundamentals/
    └── transformer-attention/
        ├── index.md
        ├── attention-flow.png
        └── qkv-example.jpg
```

这种“正文和图片放在同一个文件夹”的方式叫页面包，是 Hugo 官方推荐的内容组织方式之一。

## 4. 每个模块的复制命令

只需要修改最后一级文件夹名称。

### 普通文章

```powershell
New-Item -ItemType Directory -Path "content\posts\my-python-note"
Copy-Item "docs\templates\posts.md" "content\posts\my-python-note\index.md"
```

### AI 基础原理

```powershell
New-Item -ItemType Directory -Path "content\ai-fundamentals\my-ai-topic"
Copy-Item "docs\templates\ai-fundamentals.md" "content\ai-fundamentals\my-ai-topic\index.md"
```

### 模型推理

```powershell
New-Item -ItemType Directory -Path "content\model-inference\my-inference-topic"
Copy-Item "docs\templates\model-inference.md" "content\model-inference\my-inference-topic\index.md"
```

### 论文研读

```powershell
New-Item -ItemType Directory -Path "content\papers\paper-short-name"
Copy-Item "docs\templates\papers.md" "content\papers\paper-short-name\index.md"
```

### LeetCode

```powershell
New-Item -ItemType Directory -Path "content\leetcode\two-sum"
Copy-Item "docs\templates\leetcode.md" "content\leetcode\two-sum\index.md"
```

## 5. 模板开头的配置怎么填

每篇文章顶部两个 `---` 之间的区域称为 Front Matter：

```yaml
---
title: "文章标题"
date: 2026-07-14
draft: false
description: "用一两句话说明文章解决什么问题。"
summary: "文章归档列表中显示的简短摘要。"
tags: ["Python", "性能优化"]
categories: ["技术笔记"]
ShowToc: true
TocOpen: true
---
```

常用字段：

| 字段 | 填写方式 |
| --- | --- |
| `title` | 页面显示的文章标题 |
| `date` | `年-月-日`，例如 `2026-07-14` |
| `draft` | `false` 表示发布；`true` 表示草稿 |
| `description` | 搜索引擎和文章顶部使用的简介 |
| `summary` | 文章归档卡片中的摘要 |
| `tags` | 可填写多个关键词，使用英文逗号分隔 |
| `categories` | 文章大类，通常保留模板默认值 |
| `ShowToc` | 是否显示目录 |
| `TocOpen` | 目录是否默认展开 |

注意：英文冒号后必须有一个空格；带中文或标点的文本建议始终放在英文双引号中。

## 6. Markdown 常用写法

### 标题

```markdown
## 二级标题

### 三级标题
```

正文中不要再写一级标题 `#`，因为页面标题已经由 Front Matter 的 `title` 生成。

### 列表

```markdown
- 第一项
- 第二项

1. 第一步
2. 第二步
```

### Python 代码

````markdown
```python
def add(a: int, b: int) -> int:
    return a + b
```
````

### 图片

把 `attention-flow.png` 放到文章的 `index.md` 旁边，然后写：

```markdown
![Attention 计算流程](attention-flow.png)
```

图片文件名建议使用英文小写和连字符。图片说明应描述图片表达的内容，不要只写“图片”。

### 链接、引用和表格

```markdown
[Hugo 官方文档](https://gohugo.io/)

> 这里是一段需要强调的结论。

| 指标 | 结果 |
| --- | --- |
| 延迟 | 35 ms |
| 吞吐量 | 120 token/s |
```

## 7. 论文研读模块的额外字段

论文模板比其他模板多几个字段：

```yaml
paperStatus: "在读"
paperDomain: "模型推理"
paperVenue: "ACL 2025"
paperUrl: "https://arxiv.org/abs/xxxx.xxxxx"
```

- `paperStatus`：建议使用 `待读`、`在读`、`已读` 或 `已复现`；
- `paperDomain`：例如 `AI 基础`、`模型推理`；
- `paperVenue`：会议、期刊或 `arXiv`；
- `paperUrl`：论文原文链接。

这些字段会显示在论文研读列表卡片中。

## 8. 本地预览

Windows 可以使用官方 Winget 包安装 Hugo Extended：

```powershell
winget install Hugo.Hugo.Extended
hugo version
```

在项目根目录启动预览：

```powershell
hugo server -D
```

浏览器打开：

```text
http://localhost:1313
```

`-D` 会同时显示 `draft: true` 的草稿。修改 Markdown 并保存后，页面会自动刷新。

结束预览时，在终端按 `Ctrl + C`。

## 9. 发布前检查

- [ ] 文章位于正确的模块目录；
- [ ] 文件名为 `index.md`；
- [ ] `title`、`date`、`description` 和 `summary` 已替换；
- [ ] `draft` 已设置为 `false`；
- [ ] 图片和链接可以打开；
- [ ] Python 代码块使用了 `python` 标记；
- [ ] 本地预览没有报错；
- [ ] 没有修改 `public/`、`themes/` 或页面模板。

代码提交到 GitHub 的 `main` 分支后，Cloudflare Pages 会自动构建并发布到 <https://blog-shf.pages.dev>。

## 10. 常见问题

### 文章没有出现在文章归档中

检查 `draft` 是否为 `false`、日期是否有效，以及文章是否放在本指南列出的五个内容目录之一。

### 页面有标题但没有正文

确认正文写在第二个 `---` 之后，并且 `index.md` 使用 UTF-8 编码保存。

### 图片显示不出来

确认图片与 `index.md` 位于同一文件夹，Markdown 中的文件名与真实文件名完全一致。

### Front Matter 报错

优先检查引号、英文冒号后的空格，以及数组是否使用 `["标签一", "标签二"]` 格式。

## 官方参考

- [Hugo Windows 安装](https://gohugo.io/installation/windows/)
- [Hugo 页面包](https://gohugo.io/content-management/page-bundles/)
- [Hugo 基本命令](https://gohugo.io/getting-started/usage/)
