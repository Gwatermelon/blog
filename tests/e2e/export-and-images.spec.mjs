import fs from 'node:fs';
import { expect, test } from '@playwright/test';
import { downloadDocx, inspectDocx } from '../helpers/docx.mjs';

test('Word preserves every formula and original image in the speculative decoding article', async ({ page }, testInfo) => {
  test.setTimeout(60_000);
  await page.goto('/model-inference/speculative-decoding/');
  await page.waitForFunction(() => Boolean(window.MathJax?.startup?.promise));
  await page.evaluate(() => window.MathJax.startup.promise);
  const expected = await page.locator('.post-content').evaluate(content => ({
    formulas: content.querySelectorAll('mjx-container').length,
    images: [...content.querySelectorAll('img')].map(img => ({ name: img.alt, width: Number(img.getAttribute('width')), height: Number(img.getAttribute('height')) }))
  }));
  expect(expected.formulas).toBeGreaterThan(30);
  const { files } = await downloadDocx(page);
  const { drawings } = await inspectDocx(page, files);
  expect(drawings.filter(image => image.name.startsWith('公式：'))).toHaveLength(expected.formulas);
  expect(drawings).toHaveLength(expected.formulas + expected.images.length);
  for (const drawing of drawings) {
    expect(files.has(drawing.target)).toBe(true);
    expect(drawing.width).toBeGreaterThan(0);
    expect(drawing.height).toBeGreaterThan(0);
  }
  const formulas = drawings.filter(image => image.name.startsWith('公式：'));
  const pixels = await page.evaluate(async images => {
    const results = [];
    for (const bytes of images) {
      const bitmap = await createImageBitmap(new Blob([new Uint8Array(bytes)], { type: 'image/png' }));
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const context = canvas.getContext('2d');
      context.drawImage(bitmap, 0, 0);
      const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
      results.push(rgba.filter((value, index) => index % 4 === 3 && value > 0).length);
      bitmap.close();
    }
    return results;
  }, formulas.map(image => [...files.get(image.target)]));
  expect(pixels.every(count => count > 0)).toBe(true);
  // Keep representative generated formula images available for visual QA.
  const sample = formulas.find(image => image.name.includes('p_{\\mathrm{residual}}'));
  expect(sample).toBeTruthy();
  fs.writeFileSync(testInfo.outputPath('formula-residual.png'), files.get(sample.target));
  for (const image of expected.images) {
    const drawing = drawings.find(drawing => drawing.name === image.name);
    expect(drawing.width / drawing.height).toBeCloseTo(image.width / image.height, 3);
  }
  expect(await page.locator('.post-content mjx-container[jax="CHTML"]').count()).toBe(expected.formulas);
});

for (const slug of ['optimal-brain-surgeon', 'awq-activation-aware-weight-quantization']) {
  test(`Word preserves all formulas in ${slug}`, async ({ page }) => {
    await page.goto(`/model-inference/${slug}/`);
    await page.evaluate(() => window.MathJax.startup.promise);
    const count = await page.locator('.post-content mjx-container').count();
    expect(count).toBeGreaterThan(30);
    const { files } = await downloadDocx(page);
    const { drawings } = await inspectDocx(page, files);
    expect(drawings.filter(image => image.name.startsWith('公式：'))).toHaveLength(count);
  });
}

test('Word keeps inline, table and display formulas and restarts independent lists', async ({ page }) => {
  await page.goto('/model-inference/optimal-brain-surgeon/');
  await page.evaluate(async () => {
    await window.MathJax.startup.promise;
    const content = document.querySelector('.post-content');
    window.MathJax.typesetClear([content]);
    content.innerHTML = '<p>Before $x^2$ after</p><div>$$\\frac{a}{b}$$</div><table><tr><td>$\\sqrt{x}$</td></tr></table><ol><li>First</li><li>Second<ul><li>Nested</li></ul></li></ol><ol start="3"><li>Third</li></ol>';
    await window.MathJax.typesetPromise([content]);
  });
  const { files, document } = await downloadDocx(page);
  const structure = await inspectDocx(page, files);
  expect(structure.drawings).toHaveLength(3);
  expect(structure.tables).toBe(1);
  expect(structure.listIds[0]).toBe(structure.listIds[1]);
  expect(new Set(structure.listIds).size).toBe(3);
  expect(files.get('word/numbering.xml').toString()).toContain('<w:startOverride w:val="3"/>');
  expect(document).toMatch(/Before [\s\S]*<w:drawing>[\s\S]* after/);
  expect(document).toMatch(/<w:tbl>[\s\S]*<w:drawing>/);
});

test('article images reserve space and offer working responsive variants', async ({ page }) => {
  await page.goto('/model-inference/speculative-decoding/');
  const images = await page.locator('.post-content img').evaluateAll(images => images.map(image => ({
    width: Number(image.getAttribute('width')), height: Number(image.getAttribute('height')), srcset: image.srcset,
    renderedWidth: image.getBoundingClientRect().width, renderedHeight: image.getBoundingClientRect().height
  })));
  expect(images).toHaveLength(5);
  for (const image of images) {
    expect(image.width).toBeGreaterThan(0);
    expect(image.height).toBeGreaterThan(0);
    expect(image.renderedWidth / image.renderedHeight).toBeCloseTo(image.width / image.height, 2);
    for (const candidate of image.srcset.split(',').filter(Boolean)) {
      const response = await page.request.get(candidate.trim().split(/\s+/)[0]);
      expect(response.ok()).toBe(true);
    }
  }
  expect(images.some(image => image.srcset.includes('480w'))).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
});

test('Word reports unavailable math instead of silently exporting incomplete content', async ({ page }) => {
  await page.route('**/mathjax@3.2.2/**', route => route.abort());
  await page.goto('/model-inference/optimal-brain-surgeon/');
  await page.locator('[data-docx-download]').click();
  await expect(page.locator('[data-docx-download-status]')).toContainText('公式尚未加载完成');
  await expect(page.locator('[data-docx-download]')).toBeEnabled();
});
