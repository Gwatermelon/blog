import { expect, test } from '@playwright/test';

const api = '**/api/github-contributions?*';
const cacheKey = 'gz-github-contributions-v1:Gwatermelon';
const today = () => {
  const date = new Date();
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
};
const items = () => [{ date: today(), count: 7, level: 2 }];

test('fresh contribution cache avoids a second API request', async ({ page }) => {
  let requests = 0;
  await page.route(api, route => {
    requests++;
    return route.fulfill({ json: { contributions: items() } });
  });
  await page.goto('/');
  await expect(page.locator('[data-contribution-summary]')).toContainText('7 次贡献');
  await page.reload();
  await expect(page.locator('[data-contribution-summary]')).toContainText('7 次贡献');
  expect(requests).toBe(1);
});

test('stale contribution cache remains visible when refresh fails', async ({ page }) => {
  await page.clock.install({ time: new Date(Date.now() + 24 * 60 * 60 * 1000) });
  await page.addInitScript(({ cacheKey, items }) => {
    localStorage.setItem(cacheKey, JSON.stringify({ savedAt: Date.now() - 2 * 60 * 60 * 1000, items }));
  }, { cacheKey, items: items() });
  await page.route(api, route => route.abort());
  await page.goto('/');
  await expect(page.locator('[data-contribution-summary]')).toContainText('7 次贡献（上次记录，暂未更新）');
  await expect(page.locator('[data-contribution-calendar]')).toBeVisible();
});

test('first visit retains the bundled GitHub calendar when the API is unavailable', async ({ page }) => {
  await page.route(api, route => route.fulfill({ status: 503, json: { error: 'Unavailable' } }));
  await page.goto('/');
  await expect(page.locator('[data-contribution-summary]')).toContainText('上次记录，暂未更新');
  await expect(page.locator('[data-contribution-fallback]')).toBeHidden();
  await expect(page.locator('.github-contribution-cell').first()).toBeVisible();
  expect(await page.locator('.github-contribution-cell[data-level="4"]').count()).toBeGreaterThan(0);
});

test('invalid cache and unavailable storage do not stop a successful refresh', async ({ page }) => {
  await page.addInitScript(({ cacheKey }) => {
    localStorage.setItem(cacheKey, '{broken');
    Storage.prototype.setItem = () => { throw new Error('Storage disabled'); };
  }, { cacheKey });
  await page.route(api, route => route.fulfill({ json: { contributions: items() } }));
  await page.goto('/');
  await expect(page.locator('[data-contribution-summary]')).toContainText('7 次贡献');
});
