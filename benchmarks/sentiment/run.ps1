$ErrorActionPreference = "Stop"

$benchmarkDir = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $benchmarkDir "..\..")).Path
$environmentDir = if ($env:SENTIMENT_BENCH_ENV) {
  $env:SENTIMENT_BENCH_ENV
} else {
  Join-Path $env:LOCALAPPDATA "MLAI2026\sentiment-benchmark"
}
$python = Join-Path $environmentDir "Scripts\python.exe"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  throw "Install Python 3.13 and the Windows py launcher first."
}

if (-not (Test-Path -LiteralPath $python)) {
  New-Item -ItemType Directory -Path (Split-Path $environmentDir) -Force | Out-Null
  py -3.13 -m venv $environmentDir
  if ($LASTEXITCODE -ne 0) { throw "Could not create the Python 3.13 environment." }
}

& $python -m pip install -r (Join-Path $benchmarkDir "requirements.lock.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned benchmark dependencies." }

Push-Location $repoRoot
try {
  & $python -m unittest discover -s benchmarks/sentiment -p "test_*.py"
  if ($LASTEXITCODE -ne 0) { throw "Sentiment tests failed." }
  & $python benchmarks/sentiment/benchmark.py
  if ($LASTEXITCODE -ne 0) { throw "Sentiment benchmark failed." }
} finally {
  Pop-Location
}
