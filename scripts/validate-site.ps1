[CmdletBinding()]
param(
  [string]$PublicDir
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$issues = [System.Collections.Generic.List[string]]::new()
$mainSections = @('posts', 'ai-fundamentals', 'model-inference', 'papers', 'math', 'leetcode')

function Add-Issue([string]$Message) {
  $issues.Add($Message)
}

function Get-RelativePath([string]$BasePath, [string]$TargetPath) {
  $separator = [IO.Path]::DirectorySeparatorChar
  $baseFull = [IO.Path]::GetFullPath($BasePath).TrimEnd([char[]]@('\', '/')) + $separator
  $targetFull = [IO.Path]::GetFullPath($TargetPath)
  $baseUri = [Uri]::new($baseFull)
  $targetUri = [Uri]::new($targetFull)
  return [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString()).Replace('/', $separator)
}

function Get-FrontMatter([string]$Raw, [string]$RelativePath) {
  $match = [regex]::Match($Raw, '(?s)\A---\s*\r?\n(.*?)\r?\n---(?:\s*\r?\n|\s*\z)')
  if (-not $match.Success) {
    Add-Issue "$RelativePath has no valid YAML front matter"
    return ''
  }
  return $match.Groups[1].Value
}

function Require-Fields([string]$FrontMatter, [string]$RelativePath, [string[]]$Fields) {
  foreach ($field in $Fields) {
    if (-not [regex]::IsMatch($FrontMatter, "(?m)^$([regex]::Escape($field)):\s*\S")) {
      Add-Issue "$RelativePath is missing front matter field '$field'"
    }
  }
}

foreach ($section in $mainSections) {
  $sectionPath = Join-Path $root "content/$section"
  if (-not (Test-Path -LiteralPath $sectionPath)) { continue }

  Get-ChildItem -LiteralPath $sectionPath -Directory | ForEach-Object {
    $article = Join-Path $_.FullName 'index.md'
    if (-not (Test-Path -LiteralPath $article)) { return }
    $relative = (Get-RelativePath $root $article).Replace('\', '/')
    $raw = Get-Content -Raw -Encoding utf8 -LiteralPath $article
    $front = Get-FrontMatter $raw $relative
    Require-Fields $front $relative @(
      'title', 'date', 'lastmod', 'draft', 'description', 'summary',
      'tags', 'categories', 'ShowToc', 'TocOpen'
    )
  }
}

$specialPages = @(
  @{ Path = 'content/about/index.md'; Fields = @('title', 'date', 'lastmod', 'description', 'summary') },
  @{ Path = 'content/todo/index.md'; Fields = @('title', 'date', 'lastmod', 'description', 'robotsNoIndex') },
  @{ Path = 'content/search.md'; Fields = @('title', 'date', 'lastmod', 'description', 'summary', 'robotsNoIndex') }
)
foreach ($page in $specialPages) {
  $path = Join-Path $root $page.Path
  $raw = Get-Content -Raw -Encoding utf8 -LiteralPath $path
  $front = Get-FrontMatter $raw $page.Path
  Require-Fields $front $page.Path $page.Fields
}

$todoPath = Join-Path $root 'content/todo/index.md'
$todoRaw = Get-Content -Raw -Encoding utf8 -LiteralPath $todoPath
if (-not [regex]::IsMatch($todoRaw, '(?m)^robotsNoIndex:\s*true\s*$')) {
  Add-Issue 'content/todo/index.md must remain excluded from search-engine indexing'
}
$todoIds = [regex]::Matches($todoRaw, '(?m)^\s*-\s+id:\s*["'']?([^\r\n"'']+)') | ForEach-Object { $_.Groups[1].Value.Trim() }
$duplicateIds = $todoIds | Group-Object | Where-Object Count -gt 1
foreach ($duplicate in $duplicateIds) {
  Add-Issue "content/todo/index.md has duplicate task id '$($duplicate.Name)'"
}

Get-ChildItem -LiteralPath (Join-Path $root 'content') -Recurse -Filter *.md | ForEach-Object {
  $markdown = Get-Content -Raw -Encoding utf8 -LiteralPath $_.FullName
  $relative = (Get-RelativePath $root $_.FullName).Replace('\', '/')
  foreach ($match in [regex]::Matches($markdown, '!\[[^\]]*\]\(([^)\s]+)')) {
    $target = $match.Groups[1].Value.Trim('<', '>')
    if ($target -match '^(https?:|data:|/)') { continue }
    $imagePath = Join-Path $_.DirectoryName ([Uri]::UnescapeDataString($target))
    if (-not (Test-Path -LiteralPath $imagePath)) {
      Add-Issue "$relative references missing image '$target'"
    }
  }
}

$requiredAssets = @(
  'static/favicon.ico',
  'static/favicon-16x16.png',
  'static/favicon-32x32.png',
  'static/apple-touch-icon.png',
  'static/safari-pinned-tab.svg',
  'static/images/site-card.png'
)
foreach ($asset in $requiredAssets) {
  if (-not (Test-Path -LiteralPath (Join-Path $root $asset))) {
    Add-Issue "Required brand asset is missing: $asset"
  }
}

$config = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $root 'hugo.toml')
if ([regex]::IsMatch($config, '(?m)^\s*unsafe\s*=\s*true\s*$')) {
  Add-Issue 'hugo.toml must not enable unsafe Markdown rendering'
}

if ($PublicDir) {
  $publicRoot = [IO.Path]::GetFullPath((Join-Path $root $PublicDir))
  $requiredPages = @(
    'index.html', 'index.json', 'index.xml', 'robots.txt', 'sitemap.xml',
    'about/index.html', 'todo/index.html', 'search/index.html',
    'ai-fundamentals/add-layernorm/index.html',
    'ai-fundamentals/feedforward/index.html',
    'ai-fundamentals/self-attention/index.html',
    'ai-fundamentals/softmax-optimization/index.html',
    'papers/token-recycling/index.html',
    'math/jacobian-matrix/index.html',
    'math/taylor-series/index.html',
    'model-inference/dflash-vs-eagle3/index.html'
  )
  foreach ($page in $requiredPages) {
    if (-not (Test-Path -LiteralPath (Join-Path $publicRoot $page))) {
      Add-Issue "Generated page is missing: $page"
    }
  }

  if (Test-Path -LiteralPath $publicRoot) {
    $htmlFiles = Get-ChildItem -LiteralPath $publicRoot -Recurse -Filter *.html
    foreach ($file in $htmlFiles) {
      $html = Get-Content -Raw -Encoding utf8 -LiteralPath $file.FullName
      if ($html.Contains('0001-01-01T00:00:00')) {
        Add-Issue "$(Get-RelativePath $publicRoot $file.FullName) contains a zero structured-data date"
      }

      foreach ($match in [regex]::Matches($html, '(?i)(?:href|src)=(?:"([^"]+)"|''([^'']+)''|([^\s>]+))')) {
        $url = if ($match.Groups[1].Success) {
          $match.Groups[1].Value
        } elseif ($match.Groups[2].Success) {
          $match.Groups[2].Value
        } else {
          $match.Groups[3].Value
        }
        $url = $url.Replace('&amp;', '&')
        if ($url.StartsWith('https://blog-shf.pages.dev')) {
          $url = $url.Substring('https://blog-shf.pages.dev'.Length)
        }
        if ($url -match '^(https?:)?//' -or $url -match '^(mailto:|tel:|javascript:|data:|#)') { continue }

        $url = ($url -split '[?#]')[0]
        if (-not $url) { continue }
        $decoded = [Uri]::UnescapeDataString($url)
        $candidate = if ($decoded.StartsWith('/')) {
          Join-Path $publicRoot $decoded.TrimStart('/')
        } else {
          Join-Path $file.DirectoryName $decoded
        }
        $candidate = [IO.Path]::GetFullPath($candidate)
        if ($decoded.EndsWith('/')) {
          $candidate = Join-Path $candidate 'index.html'
        }

        $exists = Test-Path -LiteralPath $candidate
        if (-not $exists -and -not [IO.Path]::GetExtension($candidate)) {
          $exists = Test-Path -LiteralPath (Join-Path $candidate 'index.html')
        }
        if (-not $exists) {
          Add-Issue "$(Get-RelativePath $publicRoot $file.FullName) references missing local URL '$url'"
        }
      }
    }

    $homeHtml = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $publicRoot 'index.html')
    if ($homeHtml -notmatch 'site-card\.png') { Add-Issue 'Home page is missing the social preview image' }
    if ($homeHtml -notmatch '"@type":"Person"') { Add-Issue 'Home page schema publisher must be Person' }
    if ($homeHtml -notmatch 'class=github-contributions') { Add-Issue 'Home page is missing the GitHub contributions calendar' }
    if ($homeHtml -notmatch '/js/github-contributions\.min\.[a-f0-9]+\.js') { Add-Issue 'Home page is missing its fingerprinted GitHub contributions script' }
    if ($homeHtml -match 'class="first-entry home-info"') { Add-Issue 'Home page must not render the removed profile card' }

    $todo = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $publicRoot 'todo/index.html')
    if ($todo -notmatch 'noindex, nofollow') { Add-Issue 'Todo page must render noindex metadata' }
    if ($todo -notmatch '/js/todo\.min\.[a-f0-9]+\.js') { Add-Issue 'Todo page is missing its fingerprinted script' }

    $math = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $publicRoot 'math/taylor-series/index.html')
    if ($math -notmatch 'mathjax@3\.2\.2') { Add-Issue 'Math page is missing MathJax' }

    $jacobian = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $publicRoot 'math/jacobian-matrix/index.html')
    if ($jacobian -notmatch 'mathjax@3\.2\.2') { Add-Issue 'Jacobian matrix article is missing MathJax' }
    if ($jacobian -match '<h[1-6][^>]*>\$\$') { Add-Issue 'Jacobian matrix article contains a formula delimiter parsed as a heading' }

    $softmax = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $publicRoot 'ai-fundamentals/softmax-optimization/index.html')
    if ($softmax -notmatch 'mathjax@3\.2\.2') { Add-Issue 'Softmax optimization article is missing MathJax' }
    if ($softmax -notmatch 'sigmoid-function\.png') { Add-Issue 'Softmax optimization article is missing its Sigmoid figure' }

    $addLayerNorm = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $publicRoot 'ai-fundamentals/add-layernorm/index.html')
    if ($addLayerNorm -notmatch 'mathjax@3\.2\.2') { Add-Issue 'Add&LayerNorm article is missing MathJax' }
    if ($addLayerNorm -notmatch 'transformer-architecture\.png') { Add-Issue 'Add&LayerNorm article is missing its Transformer figure' }
    if ($addLayerNorm -notmatch 'residual-connection\.png') { Add-Issue 'Add&LayerNorm article is missing its residual figure' }
    if ($addLayerNorm -notmatch 'normalization-comparison\.png') { Add-Issue 'Add&LayerNorm article is missing its normalization comparison figure' }

    $feedforward = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $publicRoot 'ai-fundamentals/feedforward/index.html')
    if ($feedforward -notmatch 'mathjax@3\.2\.2') { Add-Issue 'Feedforward article is missing MathJax' }
    if ($feedforward -notmatch 'transformer-architecture\.png') { Add-Issue 'Feedforward article is missing its Transformer figure' }
    if ($feedforward -match '<h[1-6][^>]*>\$\$') { Add-Issue 'Feedforward article contains a formula delimiter parsed as a heading' }

    $selfAttention = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $publicRoot 'ai-fundamentals/self-attention/index.html')
    if ($selfAttention -notmatch 'mathjax@3\.2\.2') { Add-Issue 'Self-Attention article is missing MathJax' }
    if ($selfAttention -match '<h[1-6][^>]*>\$\$') { Add-Issue 'Self-Attention article contains a formula delimiter parsed as a heading' }
  }
}

if ($issues.Count -gt 0) {
  $issues | Sort-Object -Unique | ForEach-Object { Write-Error $_ }
  exit 1
}

Write-Output "Site validation passed ($($mainSections.Count) content sections checked)."

