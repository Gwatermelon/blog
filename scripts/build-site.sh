#!/usr/bin/env bash
set -euo pipefail

expected_version="$(tr -d '\r\n' < .hugo-version)"
candidate="${HUGO_BIN:-}"
if [[ -z "${candidate}" ]] && command -v hugo >/dev/null 2>&1; then
  candidate="$(command -v hugo)"
fi

if [[ -n "${candidate}" ]]; then
  actual_version="$("${candidate}" version 2>/dev/null || true)"
  actual_version_number="$(printf '%s' "${actual_version}" | sed -E 's/^hugo v([0-9]+\.[0-9]+\.[0-9]+).*/\1/')"
else
  actual_version="not installed"
  actual_version_number=""
fi

if [[ "${actual_version_number}" != "${expected_version}" ]]; then
  os="$(uname -s)"
  arch="$(uname -m)"
  if [[ "${os}" != "Linux" || "${arch}" != "x86_64" ]]; then
    echo "ERROR: Expected Hugo ${expected_version}, but found: ${actual_version}" >&2
    echo "Automatic Hugo installation currently supports Linux x86_64 build environments." >&2
    exit 1
  fi

  cache_root="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}}/zhangge-hugo/${expected_version}"
  candidate="${cache_root}/hugo"
  if [[ ! -x "${candidate}" ]]; then
    asset="hugo_extended_${expected_version}_linux-amd64.tar.gz"
    release="https://github.com/gohugoio/hugo/releases/download/v${expected_version}"
    download_dir="$(mktemp -d)"
    trap 'rm -rf "${download_dir}"' EXIT

    echo "Installing pinned Hugo ${expected_version} for this build."
    curl --fail --location --silent --show-error "${release}/${asset}" --output "${download_dir}/${asset}"
    curl --fail --location --silent --show-error "${release}/hugo_${expected_version}_checksums.txt" --output "${download_dir}/checksums.txt"
    (
      cd "${download_dir}"
      grep " ${asset}$" checksums.txt | sha256sum --check --strict
      tar --extract --gzip --file "${asset}" hugo
    )
    mkdir -p "${cache_root}"
    install -m 0755 "${download_dir}/hugo" "${candidate}"
  fi
fi

resolved_version="$("${candidate}" version 2>/dev/null || true)"
if [[ "${resolved_version}" != hugo\ v${expected_version}* ]]; then
  echo "ERROR: Pinned Hugo installation is unusable: ${resolved_version:-no version output}" >&2
  exit 1
fi

echo "Using ${resolved_version}"

node scripts/validate-site.mjs
"${candidate}" --cleanDestinationDir --gc --minify --panicOnWarning
node scripts/validate-site.mjs --public-dir public
