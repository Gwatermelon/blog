import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'public');
const host = '127.0.0.1';
const port = Number(process.env.PORT || 4173);
const mimeTypes = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.ico', 'image/x-icon'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.png', 'image/png'],
  ['.svg', 'image/svg+xml'],
  ['.xml', 'application/xml; charset=utf-8']
]);

function resolveRequest(requestUrl) {
  const pathname = decodeURIComponent(new URL(requestUrl, `http://${host}`).pathname);
  const relative = pathname.replace(/^\/+/, '');
  let candidate = path.resolve(root, relative);
  if (candidate !== root && !candidate.startsWith(`${root}${path.sep}`)) return null;
  if (pathname.endsWith('/')) candidate = path.join(candidate, 'index.html');
  if (!path.extname(candidate) && !fs.existsSync(candidate)) candidate = path.join(candidate, 'index.html');
  return candidate;
}

const server = http.createServer((request, response) => {
  let filePath;
  try {
    filePath = resolveRequest(request.url || '/');
  } catch {
    response.writeHead(400).end('Bad request');
    return;
  }

  if (!filePath || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
    response.writeHead(404).end('Not found');
    return;
  }

  response.writeHead(200, {
    'Content-Type': mimeTypes.get(path.extname(filePath).toLowerCase()) || 'application/octet-stream',
    'Cache-Control': 'no-store'
  });
  fs.createReadStream(filePath).pipe(response);
});

server.listen(port, host, () => {
  console.log(`Serving public/ at http://${host}:${port}`);
});
