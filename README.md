# Ge Zhang · 技术笔记

这是一个基于 [Hugo](https://gohugo.io/) 和 PaperMod 主题构建的个人技术博客，主要记录张革在 AI 基础、大模型推理、程序分析和工程实践方面的学习与思考。

## 在线访问

本项目托管在 **Cloudflare Pages**，内容持续更新并发布到以下地址：

**每日产出：<https://blog-shf.pages.dev>**

**GitHub 代码仓库：<https://github.com/Gwatermelon/blog>**

## 项目结构

```text
content/     博客文章与研究笔记
layouts/     自定义页面模板
assets/      自定义样式与脚本资源
static/      品牌图标与社交分享图片
scripts/     内容、构建产物与品牌资源校验脚本
themes/      Hugo 主题
hugo.toml    站点配置
```

## 内容写作

不熟悉 Hugo 或前端开发时，可以直接按照 [博客文章添加指南](docs/CONTENT_AUTHORING_GUIDE.md) 操作。指南提供普通文章、AI 基础原理、模型推理、论文研读和 LeetCode 五类可复制模板。

## 部署说明

代码更新后由 Cloudflare Pages 构建并发布，线上站点以 <https://blog-shf.pages.dev> 的内容为准。部署配置应保持：

- 构建命令：`hugo --cleanDestinationDir --gc --minify --panicOnWarning`
- 输出目录：`public`
- 环境变量 `HUGO_VERSION`：与根目录 `.hugo-version` 一致
- Git 子模块：启用递归拉取

`public/` 和 `.hugo_build.lock` 是本地构建产物，不纳入版本控制。

## 本地校验

```powershell
powershell -ExecutionPolicy Bypass -File scripts\validate-site.ps1
hugo --cleanDestinationDir --gc --minify --panicOnWarning
powershell -ExecutionPolicy Bypass -File scripts\validate-site.ps1 -PublicDir public
```

GitHub Actions 会在提交和 Pull Request 上执行相同检查，包括文章元数据、本地图片、内部链接、品牌资源、结构化数据和关键页面功能。

## 许可

网站程序代码使用 [MIT License](LICENSE)。原创文章与图片适用 [内容版权说明](CONTENT_LICENSE.md)，第三方材料仍归原权利人所有。

## 博客搭建原则
### 栽一棵树最好的时间是十年前，其次就是现在
### 内容尽力去保证一次性理解和记录完整
### 可以写的慢一些，但是信息要准确
### 最后相信世界是个草台班子，不必内耗，专注提升自己

