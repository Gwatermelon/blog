const username = 'Gwatermelon';
const freshFor = 60 * 60 * 1000;

const validItems = (items) => Array.isArray(items) && items.length >= 180 && items.every(item => (
  item && /^\d{4}-\d{2}-\d{2}$/.test(item.date) &&
  Number.isInteger(item.count) && item.count >= 0 &&
  Number.isInteger(item.level) && item.level >= 0 && item.level <= 4
));

// Browser requests stay on our domain; successful responses are shared at the edge.
export async function onRequestGet({ request, waitUntil }, dependencies = {}) {
  const fetchData = dependencies.fetch || fetch;
  const cache = dependencies.cache || caches.default;
  const url = new URL(request.url);
  if (url.searchParams.get('username') !== username) {
    return Response.json({ error: 'Unknown profile' }, { status: 400 });
  }
  const key = new Request(`${url.origin}/api/github-contributions?username=${username}`);
  let cached;
  try {
    const response = await cache.match(key);
    if (response) cached = await response.json();
  } catch { /* Cache availability must not prevent a refresh. */ }
  if (cached && validItems(cached.contributions) && Date.now() - cached.savedAt < freshFor) {
    return Response.json(cached, { headers: { 'Cache-Control': 'public, max-age=300' } });
  }
  try {
    const response = await fetchData(`https://github-contributions-api.jogruber.de/v4/${username}?y=last`, {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(5000)
    });
    if (!response.ok) throw new Error('Upstream unavailable');
    const data = await response.json();
    if (!validItems(data.contributions)) throw new Error('Invalid upstream data');
    const body = { contributions: data.contributions, savedAt: Date.now() };
    const stored = Response.json(body, { headers: { 'Cache-Control': 'public, max-age=604800' } });
    waitUntil(cache.put(key, stored).catch(() => {}));
    return Response.json(body, { headers: { 'Cache-Control': 'public, max-age=300' } });
  } catch {
    if (cached && validItems(cached.contributions)) {
      return Response.json({ ...cached, stale: true }, { headers: { 'Cache-Control': 'no-store' } });
    }
    // The page retains its bundled snapshot when neither upstream nor edge cache is available.
    return Response.json({ error: 'Contributions temporarily unavailable' }, { status: 503 });
  }
}
