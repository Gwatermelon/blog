import { expect, test } from '@playwright/test';
import { downloadDocx, inspectDocx } from '../helpers/docx.mjs';

test('home page keeps the contribution calendar and article feed usable', async ({ page }) => {
  await page.goto('/');

  await expect(page.locator('.home-layout')).toBeVisible();
  await expect(page.locator('.github-contributions')).toBeVisible();
  await expect(page.locator('.post-entry').first()).toBeVisible();
  await expect(page.locator('a.entry-link').first()).toHaveAttribute('href', /^https:\/\/zhangge\.dev\//);

  const hasHorizontalOverflow = await page.evaluate(() => (
    document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
  ));
  expect(hasHorizontalOverflow).toBe(false);
});

test('OBS formulas are rendered by MathJax', async ({ page }) => {
  await page.goto('/model-inference/optimal-brain-surgeon/');

  await expect(page.locator('script[src$="/tex-chtml.js"]')).toHaveAttribute('integrity', /^sha384-/);
  await expect.poll(() => page.locator('mjx-container').count(), { timeout: 15_000 }).toBeGreaterThan(5);
  await expect(page.locator('h1, h2, h3, h4, h5, h6').filter({ hasText: '$$' })).toHaveCount(0);
});

test('AWQ formulas are rendered without breaking the article layout', async ({ page }) => {
  await page.goto('/model-inference/awq-activation-aware-weight-quantization/');

  await expect(page.locator('script[src$="/tex-chtml.js"]')).toHaveAttribute('integrity', /^sha384-/);
  await expect.poll(() => page.locator('mjx-container').count(), { timeout: 15_000 }).toBeGreaterThan(30);
  await expect(page.locator('h1, h2, h3, h4, h5, h6').filter({ hasText: '$$' })).toHaveCount(0);

  const hasHorizontalOverflow = await page.evaluate(() => (
    document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
  ));
  expect(hasHorizontalOverflow).toBe(false);
});

test('Article pages expose a valid Word download', async ({ page }) => {
  const articlePath = '/model-inference/model-parallelism-deployment/';
  await page.goto(articlePath);

  const downloadButton = page.getByRole('button', { name: '下载 Word 文档' });
  await expect(downloadButton).toBeVisible();

  const { download, document, files } = await downloadDocx(page);
  expect(download.suggestedFilename()).toBe('model-parallelism-deployment.docx');

  expect(document).toContain('大模型推理并行部署');
  expect(document).toContain('Requests');
  const structure = await inspectDocx(page, files);
  expect(structure.drawings.filter(image => image.name.startsWith('公式：'))).toHaveLength(3);
  expect(structure.tables).toBeGreaterThan(0);
  expect(structure.listIds.length).toBeGreaterThan(0);
});

test('Todo completion state survives a reload', async ({ page }) => {
  await page.goto('/todo/');

  const firstTask = page.locator('[data-task-id]').first();
  await page.locator('label.todo-item').first().click();
  await expect(firstTask).toBeChecked();
  await page.reload();
  await expect(page.locator('[data-task-id]').first()).toBeChecked();
  await expect(page.locator('#todo-progress-text')).toContainText('1 /');
});
