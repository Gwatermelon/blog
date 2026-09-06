import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const expectedVersion = fs.readFileSync(path.join(root, '.hugo-version'), 'utf8').trim();
const hugo = process.env.HUGO_BIN || 'hugo';

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status || 1);
}

const version = spawnSync(hugo, ['version'], { encoding: 'utf8' });
if (version.error || version.status !== 0 || version.stdout.match(/^hugo v(\d+\.\d+\.\d+)/)?.[1] !== expectedVersion) {
  console.error(`Install Hugo ${expectedVersion} or set HUGO_BIN to its executable path.`);
  process.exit(1);
}

run(process.execPath, ['scripts/validate-site.mjs']);
run(hugo, ['--cleanDestinationDir', '--gc', '--minify', '--panicOnWarning']);
run(process.execPath, ['scripts/validate-site.mjs', '--public-dir', 'public']);
