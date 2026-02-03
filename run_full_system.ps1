# Launch Oracle 5D Full System (Server + External Tunnel)
# Wrapper to keep window open
try {
    $ServerScript = ".\server_universal.py"
    $NgrokExe = ".\ngrok.exe"

    # Check Requirements
    if (-not (Test-Path $ServerScript)) { throw "Server script not found: $ServerScript" }
    if (-not (Test-Path $NgrokExe)) { throw "Ngrok not found: $NgrokExe - Please download it." }

    # 1. Start Server
    Write-Host "Starting Oracle Server (Port 5001)..." -ForegroundColor Cyan
    Start-Process python -ArgumentList $ServerScript -WindowStyle Normal

    # 2. Wait for Server
    Start-Sleep -Seconds 2

    # 3. Start Ngrok Tunnel
    Write-Host "Starting External Tunnel..." -ForegroundColor Green
    Write-Host "Look for the URL in the new window (https://xxxx.ngrok-free.app)" -ForegroundColor Yellow
    Start-Process $NgrokExe -ArgumentList "http 5001" -WindowStyle Normal

    # 4. Open Local Client (Optional)
    Start-Process "$PSScriptRoot\juggler_oracle_client.html"

    Write-Host "`nSystem Running." -ForegroundColor Yellow
    Write-Host "1. Keep the Python window open."
    Write-Host "2. Keep the Ngrok window open."
    Write-Host "3. Copy the URL from Ngrok to your phone."
}
catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
