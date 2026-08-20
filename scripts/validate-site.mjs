import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, '..');
const issues = new Set();
const args = process.argv.slice(2);
const publicDirIndex = args.indexOf('--public-dir');
const publicDir = publicDirIndex >= 0 ? args[publicDirIndex + 1] : '';

const toPosix = (value) => value.split(path.sep).join('/');
const relativeToRoot = (value) => toPosix(path.relative(root, value));
const addIssue = (message) => issues.add(message);

function readText(filePath) {
  return fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, '');
}

function filesUnder(directory, predicate = () => true) {
  if (!fs.existsSync(directory)) return [];
  const result = [];
  const pending = [directory];

  while (pending.length > 0) {
    const current = pending.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) pending.push(fullPath);
      else if (entry.isFile() && predicate(fullPath)) result.push(fullPath);
    }
  }

  return result.sort();
}

function getFrontMatter(raw, relativePath) {
  const match = raw.match(/^---\s*\r?\n([\s\S]*?)\r?\n---(?:\s*\r?\n|\s*$)/);
  if (!match) {
    addIssue(`${relativePath} has no valid YAML front matter`);
    return '';
  }
  return match[1];
}

function requireFields(frontMatter, relativePath, fields) {
  for (const field of fields) {
    const expression = new RegExp(`^${field.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}:\\s*\\S`, 'm');
    if (!expression.test(frontMatter)) {
      addIssue(`${relativePath} is missing front matter field '${field}'`);
    }
  }
}

function stripCode(markdown) {
  return markdown
    .replace(/^(```|~~~)[^\r\n]*\r?\n[\s\S]*?^\1\s*$/gm, '')
    .replace(/`[^`\r\n]*`/g, '');
}

function containsMath(markdown) {
  const searchable = stripCode(markdown);
  return /\$\$[\s\S]*?\$\$/.test(searchable) || /(?<!\$)\$(?!\$)[^\r\n$]+(?<!\$)\$(?!\$)/.test(searchable);
}

function validateMath(markdown, relativePath) {
  const searchable = stripCode(markdown);
  const hasMath = containsMath(searchable);
  const hasMathFlag = /^math:\s*true\s*$/m.test(markdown);

  if (hasMath && !hasMathFlag) {
    addIssue(`${relativePath} contains LaTeX math but is missing 'math: true'`);
  }

  for (const match of searchable.matchAll(/\$\$([\s\S]*?)\$\$/g)) {
    if (/<[A-Za-z]/.test(match[1])) {
      addIssue(`${relativePath} contains a tag-like less-than expression inside display math; use \\lt or an equivalent notation`);
    }
    if (/^\s*[=-]\s*$/m.test(match[1])) {
      addIssue(`${relativePath} contains a standalone equals or minus line inside display math`);
    }
  }

  for (const match of searchable.matchAll(/(?<!\$)\$(?!\$)([^\r\n]*?)(?<!\$)\$(?!\$)/g)) {
    if (/<[A-Za-z]/.test(match[1])) {
      addIssue(`${relativePath} contains a tag-like less-than expression inside inline math; use \\lt or an equivalent notation`);
    }
  }

  return hasMathFlag;
}

function expectedPageForArticle(article) {
  const relative = toPosix(path.relative(path.join(root, 'content'), article));
  return relative.replace(/\/index\.md$/, '/index.html');
}

const configPath = path.join(root, 'hugo.toml');
const config = readText(configPath);
const pinnedHugoVersion = readText(path.join(root, '.hugo-version')).trim();
const baseUrlMatch = config.match(/^baseURL\s*=\s*["']([^"']+)["']\s*$/m);
const baseUrl = baseUrlMatch?.[1] ?? '';
const mainSectionsMatch = config.match(/^\s*mainSections\s*=\s*\[([^\]]+)\]\s*$/m);
const mainSections = [...(mainSectionsMatch?.[1] ?? '').matchAll(/["']([^"']+)["']/g)].map((match) => match[1]);

if (!baseUrl) addIssue('hugo.toml must define baseURL');
if (mainSections.length === 0) addIssue('hugo.toml must define params.mainSections');
if (!/^\d+\.\d+\.\d+$/.test(pinnedHugoVersion)) addIssue('.hugo-version must contain a semantic version');
if (/^\s*unsafe\s*=\s*true\s*$/m.test(config)) {
  addIssue('hugo.toml must not enable unsafe Markdown rendering');
}

const baseTemplate = readText(path.join(root, 'layouts', 'baseof.html'));
if (!baseTemplate.includes('dir="auto"') || /\.Language\.(?:Direction|LanguageDirection)/.test(baseTemplate)) {
  addIssue('layouts/baseof.html must use a Hugo-version-independent language direction');
}

const requiredArticleFields = [
  'title', 'date', 'lastmod', 'draft', 'description', 'summary',
  'tags', 'categories', 'ShowToc', 'TocOpen'
];
const expectedArticlePages = [];
const mathArticlePages = [];

for (const section of mainSections) {
  const sectionPath = path.join(root, 'content', section);
  if (!fs.existsSync(sectionPath)) {
    addIssue(`Configured content section is missing: content/${section}`);
    continue;
  }

  const markdownFiles = filesUnder(sectionPath, (file) => file.endsWith('.md') && path.basename(file) !== '_index.md');
  for (const article of markdownFiles) {
    const relative = relativeToRoot(article);
    if (path.basename(article) !== 'index.md') {
      addIssue(`${relative} must be stored as a leaf bundle named index.md`);
    }

    const raw = readText(article);
    const frontMatter = getFrontMatter(raw, relative);
    requireFields(frontMatter, relative, requiredArticleFields);

    const isDraft = /^draft:\s*true\s*$/m.test(frontMatter);
    if (path.basename(article) === 'index.md' && !isDraft) {
      const expectedPage = expectedPageForArticle(article);
      expectedArticlePages.push(expectedPage);
      if (validateMath(raw, relative)) mathArticlePages.push(expectedPage);
    } else {
      validateMath(raw, relative);
    }
  }
}

const specialPages = [
  { file: 'content/about/index.md', fields: ['title', 'date', 'lastmod', 'description', 'summary'] },
  { file: 'content/todo/index.md', fields: ['title', 'date', 'lastmod', 'description', 'robotsNoIndex'] },
  { file: 'content/search.md', fields: ['title', 'date', 'lastmod', 'description', 'summary', 'robotsNoIndex'] }
];

for (const page of specialPages) {
  const filePath = path.join(root, page.file);
  if (!fs.existsSync(filePath)) {
    addIssue(`Required content page is missing: ${page.file}`);
    continue;
  }
  const raw = readText(filePath);
  const frontMatter = getFrontMatter(raw, page.file);
  requireFields(frontMatter, page.file, page.fields);
}

const todoPath = path.join(root, 'content', 'todo', 'index.md');
if (fs.existsSync(todoPath)) {
  const todoRaw = readText(todoPath);
  if (!/^robotsNoIndex:\s*true\s*$/m.test(todoRaw)) {
    addIssue('content/todo/index.md must remain excluded from search-engine indexing');
  }
  const taskIds = [...todoRaw.matchAll(/^\s*-\s+id:\s*["']?([^\r\n"']+)/gm)].map((match) => match[1].trim());
  const seen = new Set();
  for (const id of taskIds) {
    if (seen.has(id)) addIssue(`content/todo/index.md has duplicate task id '${id}'`);
    seen.add(id);
  }
}

for (const markdownPath of filesUnder(path.join(root, 'content'), (file) => file.endsWith('.md'))) {
  const markdown = readText(markdownPath);
  const relative = relativeToRoot(markdownPath);
  validateMath(markdown, relative);

  for (const match of markdown.matchAll(/!\[[^\]]*\]\(([^)\s]+)[^)]*\)/g)) {
    const target = match[1].replace(/^<|>$/g, '');
    if (/^(https?:|data:|\/)/.test(target)) continue;
    let decodedTarget;
    try {
      decodedTarget = decodeURIComponent(target);
    } catch {
      addIssue(`${relative} contains an invalid encoded image path '${target}'`);
      continue;
    }
    const imagePath = path.resolve(path.dirname(markdownPath), decodedTarget);
    if (!fs.existsSync(imagePath)) {
      addIssue(`${relative} references missing image '${target}'`);
    }
  }
}

const requiredAssets = [
  'static/favicon.ico',
  'static/favicon-16x16.png',
  'static/favicon-32x32.png',
  'static/apple-touch-icon.png',
  'static/safari-pinned-tab.svg',
  'static/images/site-card.png'
];

for (const asset of requiredAssets) {
  if (!fs.existsSync(path.join(root, asset))) addIssue(`Required brand asset is missing: ${asset}`);
}

const cloudflareMiddlewarePath = path.join(root, 'functions', '_middleware.js');
if (!fs.existsSync(cloudflareMiddlewarePath)) {
  addIssue('Cloudflare Pages hostname redirect middleware is missing');
} else {
  const middleware = readText(cloudflareMiddlewarePath);
  if (!middleware.includes('blog-shf.pages.dev') || !middleware.includes('zhangge.dev')) {
    addIssue('Cloudflare Pages middleware must redirect the legacy hostname to the production hostname');
  }
}

if (publicDir) {
  const publicRoot = path.resolve(root, publicDir);
  if (!fs.existsSync(publicRoot)) {
    addIssue(`Generated site directory is missing: ${relativeToRoot(publicRoot)}`);
  } else {
    const requiredPages = new Set([
      'index.html', 'index.json', 'index.xml', 'robots.txt', 'sitemap.xml',
      'about/index.html', 'todo/index.html', 'search/index.html',
      ...mainSections.map((section) => `${section}/index.html`),
      ...expectedArticlePages
    ]);

    for (const page of requiredPages) {
      if (!fs.existsSync(path.join(publicRoot, page))) addIssue(`Generated page is missing: ${page}`);
    }

    let siteOrigin = '';
    if (baseUrl) {
      try {
        siteOrigin = new URL(baseUrl).origin;
      } catch {
        addIssue(`hugo.toml contains an invalid baseURL: ${baseUrl}`);
      }
    }
    const publicPrefix = `${publicRoot}${path.sep}`.toLowerCase();
    const htmlFiles = filesUnder(publicRoot, (file) => file.endsWith('.html'));

    for (const file of htmlFiles) {
      const html = readText(file);
      const relative = toPosix(path.relative(publicRoot, file));
      if (html.includes('0001-01-01T00:00:00')) addIssue(`${relative} contains a zero structured-data date`);
      if (/<h[1-6][^>]*>\s*(?:\$\$|\\)/.test(html)) addIssue(`${relative} contains a formula parsed as a heading`);
      if (/(?:<p>\s*\$\$|\$\$\s*<\/p>)/.test(html)) addIssue(`${relative} contains an unprocessed display math delimiter`);
      if (html.includes('blog-shf.pages.dev')) addIssue(`${relative} contains the legacy pages.dev domain`);

      for (const match of html.matchAll(/(?:href|src)=(?:"([^"]+)"|'([^']+)'|([^\s>]+))/gi)) {
        let url = (match[1] || match[2] || match[3]).replaceAll('&amp;', '&');
        if (/^https?:\/\//.test(url)) {
          const absoluteUrl = new URL(url);
          if (!siteOrigin || absoluteUrl.origin !== siteOrigin) continue;
          url = `${absoluteUrl.pathname}${absoluteUrl.search}${absoluteUrl.hash}`;
        }
        if (/^\/\//.test(url) || /^(mailto:|tel:|javascript:|data:|#)/.test(url)) continue;

        url = url.split(/[?#]/, 1)[0];
        if (!url) continue;
        let decoded;
        try {
          decoded = decodeURIComponent(url);
        } catch {
          addIssue(`${relative} references an invalid encoded URL '${url}'`);
          continue;
        }

        let candidate = decoded.startsWith('/')
          ? path.resolve(publicRoot, decoded.slice(1))
          : path.resolve(path.dirname(file), decoded);
        const candidateLower = candidate.toLowerCase();
        if (candidateLower !== publicRoot.toLowerCase() && !candidateLower.startsWith(publicPrefix)) {
          addIssue(`${relative} references a URL outside the generated site '${url}'`);
          continue;
        }
        if (decoded.endsWith('/')) candidate = path.join(candidate, 'index.html');

        let exists = fs.existsSync(candidate);
        if (!exists && !path.extname(candidate)) exists = fs.existsSync(path.join(candidate, 'index.html'));
        if (!exists) addIssue(`${relative} references missing local URL '${url}'`);
      }

      if (expectedArticlePages.includes(relative)) {
        if (!/data-docx-download/.test(html)) addIssue(`${relative} is missing its current-article Word download button`);
        if (!/\/js\/article-docx-download\.min\.[a-f0-9]+\.js/.test(html)) {
          addIssue(`${relative} is missing its fingerprinted current-article Word download script`);
        }
      }
    }

    for (const page of mathArticlePages) {
      const generatedPath = path.join(publicRoot, page);
      if (fs.existsSync(generatedPath)) {
        const mathHtml = readText(generatedPath);
        if (!/mathjax@3\.2\.2/.test(mathHtml)) addIssue(`${page} contains math but is missing MathJax`);
        if (!/integrity=(?:"|')?sha384-[^\s>"']+/.test(mathHtml)) addIssue(`${page} loads MathJax without an integrity hash`);
      }
    }

    const homePath = path.join(publicRoot, 'index.html');
    if (fs.existsSync(homePath)) {
      const homeHtml = readText(homePath);
      if (!/site-card\.png/.test(homeHtml)) addIssue('Home page is missing the social preview image');
      if (!/"@type":"Person"/.test(homeHtml)) addIssue('Home page schema publisher must be Person');
      if (!/class=github-contributions/.test(homeHtml)) addIssue('Home page is missing the GitHub contributions calendar');
      if (!/\/js\/github-contributions\.min\.[a-f0-9]+\.js/.test(homeHtml)) addIssue('Home page is missing its fingerprinted GitHub contributions script');
      if (/class="first-entry home-info"/.test(homeHtml)) addIssue('Home page must not render the removed profile card');
      if (!/class=home-layout/.test(homeHtml)) addIssue('Home page is missing its dedicated layout');
    }

    const todoGeneratedPath = path.join(publicRoot, 'todo', 'index.html');
    if (fs.existsSync(todoGeneratedPath)) {
      const todoHtml = readText(todoGeneratedPath);
      if (!/noindex, nofollow/.test(todoHtml)) addIssue('Todo page must render noindex metadata');
      if (!/\/js\/todo\.min\.[a-f0-9]+\.js/.test(todoHtml)) addIssue('Todo page is missing its fingerprinted script');
    }
  }
}

if (issues.size > 0) {
  for (const issue of [...issues].sort()) console.error(`ERROR: ${issue}`);
  process.exit(1);
}

console.log(`Site validation passed (${mainSections.length} content sections, ${expectedArticlePages.length} articles checked).`);
