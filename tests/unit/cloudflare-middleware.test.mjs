import assert from 'node:assert/strict';
import test from 'node:test';

import { onRequest } from '../../functions/_middleware.js';

test('legacy Pages requests permanently redirect and preserve path and query', async () => {
  let nextCalled = false;
  const response = await onRequest({
    request: new Request('https://blog-shf.pages.dev/ai-fundamentals/?page=2'),
    next: () => {
      nextCalled = true;
      return new Response('unexpected');
    }
  });

  assert.equal(response.status, 308);
  assert.equal(response.headers.get('location'), 'https://zhangge.dev/ai-fundamentals/?page=2');
  assert.equal(nextCalled, false);
});

test('production requests continue to the static asset server', async () => {
  const expected = new Response('static page', { status: 200 });
  let nextCalled = false;
  const response = await onRequest({
    request: new Request('https://zhangge.dev/model-inference/'),
    next: () => {
      nextCalled = true;
      return expected;
    }
  });

  assert.equal(response, expected);
  assert.equal(nextCalled, true);
});
