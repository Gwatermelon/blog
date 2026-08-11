# Ge Zhang · 技术笔记

这是一个基于 [Hugo](https://gohugo.io/) 和 PaperMod 主题构建的个人技术博客，主要记录张革在 AI 基础、大模型推理、程序分析和工程实践方面的学习与思考。

## 在线访问

本项目托管在 **Cloudflare Pages**，内容持续更新并发布到以下地址：

**个人域名：<https://zhangge.dev/>**

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

不熟悉 Hugo 或前端开发时，可以直接按照 [博客文章添加指南](docs/CONTENT_AUTHORING_GUIDE.md) 操作。指南提供普通文章、AI 基础原理、模型推理、论文研读、数学知识和 LeetCode 六类可复制模板。

## 部署说明

代码更新后由 Cloudflare Pages 构建并发布，线上站点以 <https://zhangge.dev/> 的内容为准。部署配置应保持：

- 构建命令：`bash scripts/build-site.sh`
- 输出目录：`public`
- Production 和 Preview 环境变量 `HUGO_VERSION`：与根目录 `.hugo-version` 一致
- Git 子模块：启用递归拉取

统一构建脚本会校验 Hugo 版本、文章元数据、公式配置、图片和内部链接。任一检查失败时必须终止部署。Cloudflare Pages 默认域名 `blog-shf.pages.dev` 应通过 Cloudflare Redirect Rule 或 Bulk Redirect 永久跳转到 `https://zhangge.dev/`；Pages 的静态 `_redirects` 文件不支持按来源域名匹配，不能用于这项整站跳转。

`public/` 和 `.hugo_build.lock` 是本地构建产物，不纳入版本控制。

## 本地校验

```powershell
powershell -ExecutionPolicy Bypass -File scripts\validate-site.ps1
hugo --cleanDestinationDir --gc --minify --panicOnWarning
powershell -ExecutionPolicy Bypass -File scripts\validate-site.ps1 -PublicDir public
```

在 Bash 环境中可以直接运行与 CI、Cloudflare 相同的完整流程：

```bash
bash scripts/build-site.sh
```

GitHub Actions 会在提交和 Pull Request 上执行相同检查，包括文章元数据、本地图片、内部链接、品牌资源、结构化数据和关键页面功能。

代码必须通过 Pull Request 合入 `main`。CI 还会使用 Playwright 在桌面与移动视口检查首页、OBS 公式渲染和 Todo 状态持久化。每日生产验收会检查自定义域名、关键文章以及 `pages.dev` 永久跳转；该检查失败通常表示 Cloudflare 控制台配置与仓库要求发生了漂移。

本地运行浏览器测试：

```powershell
npm ci
npx playwright install chromium
npm run test:e2e
```

## 许可

网站程序代码使用 [MIT License](LICENSE)。原创文章与图片适用 [内容版权说明](CONTENT_LICENSE.md)，第三方材料仍归原权利人所有。

## 博客搭建原则
### 栽一棵树最好的时间是十年前，其次就是现在
### 内容尽力去保证一次性理解和记录完整
### 可以写的慢一些，但是信息要准确
### 最后相信世界是个草台班子，不必内耗，专注提升自己

