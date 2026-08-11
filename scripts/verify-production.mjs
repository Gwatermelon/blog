const productionOrigin = 'https://zhangge.dev';
const legacyOrigin = 'https://blog-shf.pages.dev';
const issues = [];

async function get(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15_000);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

const pages = ['/', '/model-inference/optimal-brain-surgeon/'];
for (const pathname of pages) {
  try {
    const response = await get(`${productionOrigin}${pathname}`);
    const html = await response.text();
    if (!response.ok) issues.push(`${pathname} returned HTTP ${response.status}`);
    if (response.url !== `${productionOrigin}${pathname}`) issues.push(`${pathname} resolved to ${response.url}`);
    if (html.includes('blog-shf.pages.dev')) issues.push(`${pathname} contains the legacy domain`);

    if (pathname.includes('optimal-brain-surgeon')) {
      if (!html.includes('mathjax@3.2.2')) issues.push(`${pathname} is missing MathJax`);
      if (!/integrity=(?:"|')?sha384-/.test(html)) issues.push(`${pathname} loads MathJax without SRI`);
      if (/(?:<p>\s*\$\$|\$\$\s*<\/p>)/.test(html)) issues.push(`${pathname} contains raw display-math delimiters`);
    }
  } catch (error) {
    issues.push(`${pathname} could not be fetched: ${error.message}`);
  }
}

try {
  const pathname = '/ai-fundamentals/';
  const response = await get(`${legacyOrigin}${pathname}`, { redirect: 'manual' });
  const location = response.headers.get('location') || '';
  if (![301, 308].includes(response.status)) {
    issues.push(`Legacy domain returned HTTP ${response.status}; expected a permanent redirect`);
  } else if (new URL(location, legacyOrigin).href !== `${productionOrigin}${pathname}`) {
    issues.push(`Legacy domain redirects to '${location}' instead of preserving the path on ${productionOrigin}`);
  }
} catch (error) {
  issues.push(`Legacy domain could not be checked: ${error.message}`);
}

if (issues.length > 0) {
  for (const issue of issues) console.error(`ERROR: ${issue}`);
  process.exit(1);
}

console.log('Production verification passed.');
