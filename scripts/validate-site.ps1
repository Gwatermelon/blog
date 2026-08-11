[CmdletBinding()]
param(
  [string]$PublicDir
)

$ErrorActionPreference = 'Stop'
$arguments = @((Join-Path $PSScriptRoot 'validate-site.mjs'))
if ($PublicDir) {
  $arguments += @('--public-dir', $PublicDir)
}

& node @arguments
exit $LASTEXITCODE
