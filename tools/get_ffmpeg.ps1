$ErrorActionPreference = 'Stop'

param(
  [string]$Url = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$zipPath = Join-Path $here 'ffmpeg.zip'

Write-Host "Downloading FFmpeg..." -ForegroundColor Cyan
Write-Host "- URL: $Url"
Write-Host "- To : $zipPath"

Invoke-WebRequest -Uri $Url -OutFile $zipPath

Write-Host "Extracting..." -ForegroundColor Cyan
$tmp = Join-Path $here '_ffmpeg_tmp'
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory -Path $tmp | Out-Null

Expand-Archive -Path $zipPath -DestinationPath $tmp -Force

$ffmpegExe = Get-ChildItem -Path $tmp -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1
$ffprobeExe = Get-ChildItem -Path $tmp -Recurse -Filter 'ffprobe.exe' | Select-Object -First 1

if (-not $ffmpegExe) {
  throw "ffmpeg.exe not found in downloaded archive"
}

Copy-Item -Force $ffmpegExe.FullName (Join-Path $here 'ffmpeg.exe')
if ($ffprobeExe) {
  Copy-Item -Force $ffprobeExe.FullName (Join-Path $here 'ffprobe.exe')
}

Write-Host "Cleaning up..." -ForegroundColor Cyan
Remove-Item -Force $zipPath
Remove-Item -Recurse -Force $tmp

Write-Host "Done." -ForegroundColor Green
Write-Host "- Installed: tools\\ffmpeg.exe"
if (Test-Path (Join-Path $here 'ffprobe.exe')) {
  Write-Host "- Installed: tools\\ffprobe.exe"
}
