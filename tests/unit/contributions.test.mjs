import assert from 'node:assert/strict';
import test from 'node:test';
import { onRequestGet } from '../../functions/api/github-contributions.js';

const contributions = Array.from({ length: 366 }, (_, i) => ({
  date: new Date(Date.UTC(2025, 10, 9 + i)).toISOString().slice(0, 10), count: 1, level: 1
}));
const request = new Request('https://zhangge.dev/api/github-contributions?username=Gwatermelon');

test('contribution endpoint caches a valid response and reuses it without upstream', async () => {
  let stored;
  const jobs = [];
  const cache = { match: async () => stored?.clone(), put: async (_, value) => { stored = value; } };
  const context = { request, waitUntil: promise => jobs.push(promise) };
  const response = await onRequestGet(context, { cache, fetch: async () => Response.json({ contributions }) });
  assert.equal(response.status, 200);
  await Promise.all(jobs);
  const reused = await onRequestGet(context, { cache, fetch: () => { throw new Error('Should use cache'); } });
  assert.deepEqual((await reused.json()).contributions, contributions);
});

test('contribution endpoint retains stale data on upstream 504 or malformed data', async () => {
  const snapshot = { contributions, savedAt: Date.now() - 7200000 };
  const cache = { match: async () => Response.json(snapshot) };
  for (const response of [new Response('', { status: 504 }), Response.json({ contributions: [] })]) {
    const result = await onRequestGet({ request, waitUntil() {} }, { cache, fetch: async () => response });
    const data = await result.json();
    assert.equal(data.stale, true);
    assert.equal(data.savedAt, snapshot.savedAt);
    assert.deepEqual(data.contributions, contributions);
  }
});

test('contribution endpoint reports unavailable data without a cache and rejects other profiles', async () => {
  const dependencies = { cache: { match: async () => undefined }, fetch: async () => { throw new Error('Network'); } };
  assert.equal((await onRequestGet({ request, waitUntil() {} }, dependencies)).status, 503);
  assert.equal((await onRequestGet({ request: new Request('https://zhangge.dev/api/github-contributions?username=other') }, dependencies)).status, 400);
});
