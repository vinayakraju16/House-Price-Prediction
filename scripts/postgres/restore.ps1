[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BackupName,

    [switch]$ConfirmDatabaseReset
)

$ErrorActionPreference = 'Stop'

if (-not $ConfirmDatabaseReset) {
    throw 'Restore replaces objects in the configured database. Re-run with -ConfirmDatabaseReset.'
}
if ($BackupName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*\.dump$') {
    throw 'BackupName must be a simple .dump filename without directories.'
}

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$backupDirectory = (Resolve-Path -LiteralPath (Join-Path $repositoryRoot 'backups\postgres')).Path
$backupPath = (Resolve-Path -LiteralPath (Join-Path $backupDirectory $BackupName)).Path
if ((Split-Path -Parent $backupPath) -ne $backupDirectory) {
    throw "Backup must be located directly inside $backupDirectory"
}

$containerCommand = 'pg_restore --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" ' +
    '--clean --if-exists --no-owner --no-privileges --exit-on-error "/backups/' + $BackupName + '"'

Push-Location $repositoryRoot
try {
    & docker compose -f docker-compose.yml -f docker-compose.postgres.yml exec -T database `
        sh -c $containerCommand
    if ($LASTEXITCODE -ne 0) {
        throw "pg_restore failed with exit code $LASTEXITCODE."
    }
    Write-Output "Restored $BackupName into the configured PostgreSQL database."
}
finally {
    Pop-Location
}
