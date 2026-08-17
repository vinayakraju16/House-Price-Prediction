[CmdletBinding()]
param(
    [string]$OutputName = "house-price-$((Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')).dump"
)

$ErrorActionPreference = 'Stop'

if ($OutputName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*\.dump$') {
    throw 'OutputName must be a simple .dump filename without directories.'
}

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$backupDirectory = Join-Path $repositoryRoot 'backups\postgres'
$containerCommand = 'pg_dump --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" ' +
    '--format custom --no-owner --no-privileges --file "/backups/' + $OutputName + '"'

Push-Location $repositoryRoot
try {
    & docker compose -f docker-compose.yml -f docker-compose.postgres.yml exec -T database `
        sh -c $containerCommand
    if ($LASTEXITCODE -ne 0) {
        throw "pg_dump failed with exit code $LASTEXITCODE."
    }

    $backupPath = Join-Path $backupDirectory $OutputName
    if (-not (Test-Path -LiteralPath $backupPath -PathType Leaf)) {
        throw "PostgreSQL reported success but the backup file was not created: $backupPath"
    }
    Get-Item -LiteralPath $backupPath | Select-Object FullName, Length, LastWriteTimeUtc
}
finally {
    Pop-Location
}
