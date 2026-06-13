# load_env.ps1 — Charge les variables d'environnement depuis .env
# Usage : . .\load_env.ps1   (le point au début est important !)

$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Error ".env introuvable : $envFile"
    return
}

Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    # Ignorer les commentaires et lignes vides
    if ($line -and $line -notmatch '^\s*#') {
        if ($line -match '^([^=]+)=(.*)$') {
            $key   = $Matches[1].Trim()
            $value = $Matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value, 'Process')
            Write-Host "  $key = $value" -ForegroundColor Green
        }
    }
}
Write-Host "`n.env chargé ✅" -ForegroundColor Cyan
