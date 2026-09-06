import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
// Linux CI installs pinned Hugo through the deployment script when needed.
const command = process.platform === 'linux' ? 'bash' : process.execPath;
const args = process.platform === 'linux' ? ['scripts/build-site.sh'] : ['scripts/build-site.mjs'];
const result = spawnSync(command, args, { cwd: root, stdio: 'inherit' });
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status || 1);
// Starting the server only after a successful build also works with no public/ yet.
await import('./serve-public.mjs');
