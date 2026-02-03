# Run Island Profiler (CUDA Edition)
# Usage: ./run_cuda_simulation.ps1 [Trials] [MachineCode]
# If no arguments are provided, it launches in Interactive Mode.

param(
    [long]$Trials = 0,     # If 0, ask user
    [string]$Machine = ""   # If empty, ask user
)

$ExePath = ".\cuda_profiler_v7.exe"
$BaseOutFile = "matrix_db_5d"

if (-not (Test-Path $ExePath)) {
    Write-Host "Executable not found at $ExePath." -ForegroundColor Red
    Write-Host "Please build the project first." -ForegroundColor Red
    exit 1
}

# --- Safety Check: Ensure Server is Closed ---
$ServerProcess = Get-Process python -ErrorAction SilentlyContinue
if ($ServerProcess) {
    Write-Host "`n[WARNING] Python process detected!" -ForegroundColor Red -BackgroundColor Black
    Write-Host "If 'launch_oracle.ps1' (Server) is running, the DATA FILE IS LOCKED." -ForegroundColor Yellow
    Write-Host "The simulation CANNOT save data while the server is open." -ForegroundColor Yellow
    Write-Host "Please CLOS E the Server window before continuing." -ForegroundColor White
    $cont = Read-Host "Press [Enter] after closing the server (or type 'ignore' to risk it)"
    if ($cont -ne "ignore") {
        # Check again
        if (Get-Process python -ErrorAction SilentlyContinue) {
            Write-Host "Python is still running. Proceeding at your own risk..." -ForegroundColor DarkGray
        }
    }
}

# --- Interactive Menu: Select Machine ---
if ($Machine -eq "") {
    Clear-Host
    Write-Host "=== Juggler Island Simulation (CUDA RTX 5060 Ti) ===" -ForegroundColor Cyan
    Write-Host "Select Target Machine:" -ForegroundColor White
    Write-Host "[1] My Juggler V (my)" -ForegroundColor Green
    Write-Host "[2] I'm Juggler EX (im)" -ForegroundColor Yellow
    Write-Host "[3] Funky Juggler 2 (funky)" -ForegroundColor Magenta
    Write-Host "[4] Happy Juggler VIII (happy)" -ForegroundColor Cyan
    Write-Host "[5] Gogo Juggler 3 (gogo)" -ForegroundColor White
    Write-Host "[6] Mr. Juggler (mr)" -ForegroundColor Red
    Write-Host "[7] Ultra Miracle Juggler (ultra)" -ForegroundColor Blue
    Write-Host "[8] ALL MACHINES (Sequential)" -ForegroundColor DarkMagenta
    
    $selection = Read-Host "Enter Number or Code"
    
    switch ($selection) {
        "1" { $Machine = "my" }
        "my" { $Machine = "my" }
        "2" { $Machine = "im" }
        "im" { $Machine = "im" }
        "3" { $Machine = "funky" }
        "funky" { $Machine = "funky" }
        "4" { $Machine = "happy" }
        "happy" { $Machine = "happy" }
        "5" { $Machine = "gogo" }
        "gogo" { $Machine = "gogo" }
        "6" { $Machine = "mr" }
        "mr" { $Machine = "mr" }
        "7" { $Machine = "ultra" }
        "ultra" { $Machine = "ultra" }
        "8" { $Machine = "all" }
        "all" { $Machine = "all" }
        default { 
            Write-Host "Invalid Selection. Defaulting to My Juggler (my)." -ForegroundColor Yellow
            $Machine = "my"
        }
    }
}

# --- Interactive Menu: Select Trials ---
if ($Trials -eq 0) {
    Write-Host "`nSelect Simulation Depth (Unit Selection):" -ForegroundColor Cyan
    Write-Host "Note: 1 Island Event = Approx 140,000 Games (20 Machines x 7000G)" -ForegroundColor Gray
    Write-Host "[1] 1 Trillion GAMES (1兆ゲーム) - Recommended (~2 sec)" -ForegroundColor Green
    Write-Host "[2] 100 Trillion GAMES (100兆ゲーム) - High Precision (~3 mins)" -ForegroundColor Yellow
    Write-Host "[3] 1 Billion ISLAND EVENTS (10億アイランド) - ~2.5 mins" -ForegroundColor Magenta
    Write-Host "[4] 1 Trillion ISLAND EVENTS (1兆アイランド) - Extreme (~40 Hours)" -ForegroundColor Red
    Write-Host "[5] Custom Input (Enter raw Trial count)" -ForegroundColor Gray
    
    $t_select = Read-Host "Enter Number"
    
    switch ($t_select) {
        "1" { 
            # 1 Trillion Games / 140,000 Games per Event = ~7,142,857 Events
            $Trials = 7142857 
            Write-Host ">> Selected: 1 Trillion Games (Running 7,142,857 Events)" -ForegroundColor Green
        }
        "2" { 
            # 100 Trillion Games / 140,000 = ~714,285,714 Events
            $Trials = 714285714
            Write-Host ">> Selected: 100 Trillion Games (Running 714,285,714 Events)" -ForegroundColor Yellow
        }
        "3" { 
            $Trials = 1000000000 
            Write-Host ">> Selected: 1 Billion Events (Running 140 Trillion Games)" -ForegroundColor Magenta
        }
        "4" { 
            $Trials = 1000000000000 
            Write-Host ">> Selected: 1 Trillion Events (Running 140 Quadrillion Games)" -ForegroundColor Red
            Write-Host "WARNING: This will take approx 40 hours." -ForegroundColor Red
            # Pause removed for batch run
        }
        default {
            Write-Host "Enter Custom Count (Raw Events):" -ForegroundColor Cyan
            $inputTrials = Read-Host "Trials"
            if ($inputTrials -match '^\d+$') {
                $Trials = [long]$inputTrials
            }
            else {
                Write-Host "Invalid Input. Defaulting to 1 Billion Events." -ForegroundColor Yellow
                $Trials = 1000000000
            }
        }
    }
}

# --- Execution ---
Clear-Host
Write-Host "=== Island Profiler: CUDA Edition ===" -ForegroundColor Cyan
Write-Host "Target: RTX 5060 Ti (Dual Grid VRAM Engine)" -ForegroundColor Green
Write-Host "Trials (Events): $Trials" -ForegroundColor Yellow

$MachinesToRun = @()
if ($Machine -eq "all") {
    $MachinesToRun = @("my", "im", "funky", "happy", "gogo", "mr", "ultra")
    Write-Host ">> Batch Mode: Running ALL Machines Sequentially" -ForegroundColor Magenta
}
else {
    $MachinesToRun = @($Machine)
}

$GlobalStart = Get-Date

foreach ($m in $MachinesToRun) {
    Write-Host "`n>>> Starting Simulation for: $m" -ForegroundColor White -BackgroundColor DarkBlue
    $CurrentOutFile = "${BaseOutFile}_${m}_cuda.bin"
    
    # Measure Time
    $StartDate = Get-Date

    # --- Check for Existing File & Resume ---
    $ModeArg = "overwrite"
    if (Test-Path $CurrentOutFile) {
        if ($Machine -eq "all") {
            # Default to RESUME in batch mode to preserve successful runs if re-run
            $ModeArg = "resume"
            Write-Host ">> File exists. RESUMING (adding trials)." -ForegroundColor Cyan
        }
        else {
            Write-Host "File '$CurrentOutFile' already exists." -ForegroundColor Yellow
            $choice = Read-Host "[R]esume (Add data) or [O]verwrite (Start New)? [Default: R]"
            if ($choice -eq "" -or $choice -match "^[Rr]") { 
                $ModeArg = "resume" 
            }
            else {
                $ModeArg = "overwrite"
            }
        }
    }

    # Run Simulation
    Write-Host "Running CUDA Simulation for: $m ($ModeArg)" -ForegroundColor Green
    & $ExePath $Trials $m $CurrentOutFile $ModeArg

    $EndDate = Get-Date
    $Duration = $EndDate - $StartDate

    Write-Host "Finished $m in $($Duration.TotalSeconds) seconds." -ForegroundColor Cyan
    
    # Verify Output
    if (Test-Path $CurrentOutFile) {
        $Size = (Get-Item $CurrentOutFile).Length
        $SizeGB = "{0:N2}" -f ($Size / 1GB)
        Write-Host "Output: $CurrentOutFile ($SizeGB GB)" -ForegroundColor Green
    }
    else {
        Write-Error "Output file not generated for $m."
    }
}

$GlobalEnd = Get-Date
$GlobalDuration = $GlobalEnd - $GlobalStart

Write-Host "`n========================================" -ForegroundColor DarkGray
Write-Host "ALL TASKS COMPLETED." -ForegroundColor Green
Write-Host "Total Batch Time: $($GlobalDuration.TotalSeconds) seconds" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor DarkGray

Write-Host "`nPress Enter to exit..." -NoNewline
$null = Read-Host
