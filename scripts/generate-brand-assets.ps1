[CmdletBinding()]
param(
  [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

if (-not $OutputDirectory) {
  $OutputDirectory = Join-Path $PSScriptRoot '..\static'
}

$output = [IO.Path]::GetFullPath($OutputDirectory)
$images = Join-Path $output 'images'
New-Item -ItemType Directory -Force -Path $output, $images | Out-Null

function New-Canvas([int]$Width, [int]$Height) {
  $bitmap = [Drawing.Bitmap]::new($Width, $Height, [Drawing.Imaging.PixelFormat]::Format32bppArgb)
  $bitmap.SetResolution(96, 96)
  return $bitmap
}

function New-BrandFont([float]$Size, [Drawing.FontStyle]$Style = [Drawing.FontStyle]::Regular) {
  return [Drawing.Font]::new('Microsoft YaHei UI', $Size, $Style, [Drawing.GraphicsUnit]::Pixel)
}

function Initialize-Graphics([Drawing.Bitmap]$Bitmap) {
  $graphics = [Drawing.Graphics]::FromImage($Bitmap)
  $graphics.SmoothingMode = [Drawing.Drawing2D.SmoothingMode]::AntiAlias
  $graphics.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
  $graphics.PixelOffsetMode = [Drawing.Drawing2D.PixelOffsetMode]::HighQuality
  $graphics.TextRenderingHint = [Drawing.Text.TextRenderingHint]::AntiAliasGridFit
  return $graphics
}

function Write-Favicon([int]$Size, [string]$Path) {
  $bitmap = New-Canvas $Size $Size
  $graphics = Initialize-Graphics $bitmap
  $navy = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml('#14213d'))
  $blue = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml('#2868d8'))
  $teal = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml('#20b99a'))
  $white = [Drawing.SolidBrush]::new([Drawing.Color]::White)
  $font = New-BrandFont ($Size * 0.34) ([Drawing.FontStyle]::Bold)
  $format = [Drawing.StringFormat]::new()
  $format.Alignment = [Drawing.StringAlignment]::Center
  $format.LineAlignment = [Drawing.StringAlignment]::Center

  try {
    $graphics.FillRectangle($navy, 0, 0, $Size, $Size)
    $graphics.FillRectangle($blue, 0, 0, [Math]::Max(2, $Size * 0.12), $Size)
    $graphics.FillEllipse($teal, $Size * 0.73, $Size * 0.12, $Size * 0.13, $Size * 0.13)
    $graphics.DrawString('GZ', $font, $white, [Drawing.RectangleF]::new(0, 0, $Size, $Size), $format)
    $bitmap.Save($Path, [Drawing.Imaging.ImageFormat]::Png)
  } finally {
    $format.Dispose()
    $font.Dispose()
    $white.Dispose()
    $teal.Dispose()
    $blue.Dispose()
    $navy.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()
  }
}

function Write-PngIco([string]$PngPath, [string]$IcoPath, [byte]$Size) {
  $png = [IO.File]::ReadAllBytes($PngPath)
  $stream = [IO.File]::Create($IcoPath)
  $writer = [IO.BinaryWriter]::new($stream)
  try {
    $writer.Write([uint16]0)
    $writer.Write([uint16]1)
    $writer.Write([uint16]1)
    $writer.Write($Size)
    $writer.Write($Size)
    $writer.Write([byte]0)
    $writer.Write([byte]0)
    $writer.Write([uint16]1)
    $writer.Write([uint16]32)
    $writer.Write([uint32]$png.Length)
    $writer.Write([uint32]22)
    $writer.Write($png)
  } finally {
    $writer.Dispose()
    $stream.Dispose()
  }
}

function Write-SocialCard([string]$Path) {
  $bitmap = New-Canvas 1200 630
  $graphics = Initialize-Graphics $bitmap
  $background = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml('#0c1220'))
  $panel = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml('#14213d'))
  $blue = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml('#70a5ff'))
  $teal = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml('#20b99a'))
  $white = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml('#edf3ff'))
  $muted = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml('#9eabc2'))
  $titleFont = New-BrandFont 68 ([Drawing.FontStyle]::Bold)
  $subtitleFont = New-BrandFont 38 ([Drawing.FontStyle]::Bold)
  $bodyFont = New-BrandFont 22
  $smallFont = New-BrandFont 20
  $logoFont = New-BrandFont 32 ([Drawing.FontStyle]::Bold)
  $center = [Drawing.StringFormat]::new()
  $center.Alignment = [Drawing.StringAlignment]::Center
  $center.LineAlignment = [Drawing.StringAlignment]::Center

  try {
    $graphics.FillRectangle($background, 0, 0, 1200, 630)
    $graphics.FillRectangle($panel, 0, 0, 1200, 12)
    $graphics.FillRectangle($blue, 0, 0, 18, 630)
    $graphics.FillRectangle($panel, 825, 0, 375, 630)
    $graphics.FillRectangle($teal, 825, 0, 10, 630)
    $graphics.FillRectangle($blue, 72, 68, 86, 86)
    $graphics.DrawString('GZ', $logoFont, $white, [Drawing.RectangleF]::new(72, 68, 86, 86), $center)
    $graphics.DrawString('Ge Zhang', $titleFont, $white, 72, 205)
    $graphics.DrawString('TECHNICAL NOTES', $subtitleFont, $blue, 76, 294)
    $graphics.DrawString('AI FUNDAMENTALS  /  LLM INFERENCE', $bodyFont, $muted, 76, 374)
    $graphics.DrawString('PROGRAM ANALYSIS  /  KERNEL OPTIMIZATION', $bodyFont, $muted, 76, 412)
    $graphics.DrawString('blog-shf.pages.dev', $smallFont, $muted, 76, 520)

    $graphics.FillRectangle($blue, 905, 118, 180, 18)
    $graphics.FillRectangle($teal, 905, 174, 118, 18)
    $graphics.FillRectangle($white, 905, 230, 224, 18)
    $graphics.FillRectangle($blue, 905, 342, 224, 18)
    $graphics.FillRectangle($teal, 905, 398, 155, 18)
    $graphics.FillRectangle($white, 905, 454, 190, 18)

    $bitmap.Save($Path, [Drawing.Imaging.ImageFormat]::Png)
  } finally {
    $center.Dispose()
    $logoFont.Dispose()
    $smallFont.Dispose()
    $bodyFont.Dispose()
    $subtitleFont.Dispose()
    $titleFont.Dispose()
    $muted.Dispose()
    $white.Dispose()
    $teal.Dispose()
    $blue.Dispose()
    $panel.Dispose()
    $background.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()
  }
}

$favicon16 = Join-Path $output 'favicon-16x16.png'
$favicon32 = Join-Path $output 'favicon-32x32.png'
Write-Favicon 16 $favicon16
Write-Favicon 32 $favicon32
Write-Favicon 180 (Join-Path $output 'apple-touch-icon.png')
Write-PngIco $favicon32 (Join-Path $output 'favicon.ico') 32
Write-SocialCard (Join-Path $images 'site-card.png')

Write-Output "Brand assets generated in $output"
