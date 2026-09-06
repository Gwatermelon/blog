function withTimeout(promise, milliseconds = 15_000) {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('公式加载超时，请检查网络后重试')), milliseconds);
    })
  ]).finally(() => clearTimeout(timer));
}

async function rasterize(svg, width, height) {
  svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  svg.setAttribute('width', String(width));
  svg.setAttribute('height', String(height));
  svg.style.color = '#000000';
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], { type: 'image/svg+xml' }));
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const canvas = document.createElement('canvas');
    // Three device pixels per document pixel keep fractions and small scripts legible.
    const scale = Math.min(3, 8192 / Math.max(width, height));
    canvas.width = Math.max(1, Math.ceil(width * scale));
    canvas.height = Math.max(1, Math.ceil(height * scale));
    const context = canvas.getContext('2d');
    if (!context) throw new Error('浏览器无法生成公式图片');
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('公式图片生成失败');
    return new Uint8Array(await blob.arrayBuffer());
  } finally {
    URL.revokeObjectURL(url);
  }
}

// Keep the page's CHTML output intact. A separate MathML -> SVG document reuses
// the already parsed math, preserving macros and avoiding duplicate equation counters.
// This adapter targets the site's pinned MathJax 3.2.2 runtime.
export async function prepareMath(content, addDrawing) {
  const mj = window.MathJax;
  const runs = new WeakMap();
  if (!mj) return runs;
  if (!mj.startup?.promise) throw new Error('公式尚未加载完成，请稍后重试');
  await withTimeout(mj.startup.promise);
  const items = mj.startup.document.getMathItemsWithin(content);
  if (!items.length) return runs;
  await withTimeout(mj.loader.load('input/mml', 'output/svg'));
  const engine = mj._.mathjax.mathjax;
  const mathDocument = engine.document(document, {
    InputJax: new mj.startup.constructors.mml(mj.config.mml),
    OutputJax: new mj.startup.constructors.svg({ ...mj.config.svg, fontCache: 'none' }),
    enableMenu: false,
    enableAssistiveMml: false
  });
  const ex = 22 / 3; // 11 pt document text, expressed in CSS pixels.
  for (const item of items) {
    const converted = await engine.handleRetriesFor(() => mathDocument.convert(mj.startup.toMML(item.root), {
      format: 'MathML', display: item.display, em: ex * 2, ex, containerWidth: 700
    }));
    const svg = converted.querySelector('svg');
    if (!svg || svg.querySelector('[data-mml-node="merror"]')) throw new Error('公式无法完整转换，请检查文章公式');
    const viewBox = svg.viewBox.baseVal;
    const width = parseFloat(svg.getAttribute('width')) * ex;
    const height = width * viewBox.height / viewBox.width;
    if (!(width > 0 && height > 0)) throw new Error('公式图片尺寸无效');
    const position = item.display ? 0 : Math.round((parseFloat(svg.style.verticalAlign) || 0) * ex * 1.5);
    const data = await rasterize(svg, width, height);
    runs.set(item.typesetRoot, addDrawing({
      data, contentType: 'image/png', width, height, name: `公式：${item.math}`, position
    }));
  }
  return runs;
}
