#!/usr/bin/env bash
set -euo pipefail

expected_version="$(tr -d '\r\n' < .hugo-version)"
actual_version="$(hugo version)"
actual_version_number="$(printf '%s' "${actual_version}" | sed -E 's/^hugo v([0-9]+\.[0-9]+\.[0-9]+).*/\1/')"

if [[ "${actual_version_number}" != "${expected_version}" ]]; then
  echo "ERROR: Expected Hugo ${expected_version}, but found: ${actual_version}" >&2
  echo "Set HUGO_VERSION=${expected_version} in both Cloudflare Pages environments." >&2
  exit 1
fi

node scripts/validate-site.mjs
hugo --cleanDestinationDir --gc --minify --panicOnWarning
node scripts/validate-site.mjs --public-dir public
