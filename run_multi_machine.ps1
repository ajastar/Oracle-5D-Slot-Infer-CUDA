$ErrorActionPreference = "Stop"

function Show-Menu {
    Clear-Host
    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host "   THE ORACLE 5D - MULTI-MACHINE SELECTOR    " -ForegroundColor Cyan
    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host "1. My Juggler V (User Spec)"
    Write-Host "2. GoGo Juggler 3 (User Spec)"
    Write-Host "3. Happy Juggler V III (User Spec)"
    Write-Host "4. Funky Juggler 2 (User Spec)"
    Write-Host "5. Ultra Miracle Juggler (User Spec)"
    Write-Host "6. Mr. Juggler (User Spec)"
    Write-Host "7. S Im Juggler EX (User Spec)"
    Write-Host "Q. Quit"
    Write-Host "---------------------------------------------"
}

do {
    Show-Menu
    $choice = Read-Host "Select Machine to Simulate (1-7)"
    
    $machine = ""
    $outfile = ""
    $name = ""

    switch ($choice) {
        "1" { $machine = "my"; $outfile = "matrix_db_myjug.bin"; $name = "My Juggler V" }
        "2" { $machine = "gogo"; $outfile = "matrix_db_gogo.bin"; $name = "GoGo Juggler 3" }
        "3" { $machine = "happy"; $outfile = "matrix_db_happy.bin"; $name = "Happy Juggler V III" }
        "4" { $machine = "funky"; $outfile = "matrix_db_funky.bin"; $name = "Funky Juggler 2" }
        "5" { $machine = "ultra"; $outfile = "matrix_db_ultra.bin"; $name = "Ultra Miracle Juggler" }
        "6" { $machine = "mr"; $outfile = "matrix_db_mr.bin"; $name = "Mr. Juggler" }
        "7" { $machine = "im"; $outfile = "matrix_db_5d.bin"; $name = "S Im Juggler EX" }
        "Q" { exit }
        default { Write-Host "Invalid selection. Press Enter." -ForegroundColor Red; Read-Host; continue }
    }

    Write-Host "`nSelected: $name" -ForegroundColor Green
    Write-Host "Output File: $outfile" -ForegroundColor Gray
    
    $run = Read-Host "Start 1 Trillion Simulations? (Y/N)"
    if ($run -eq "Y" -or $run -eq "y") {
        Write-Host "Starting Simulation... (Target: 1 Trillion)" -ForegroundColor Yellow
        $start = Get-Date
        
        # Args: [trials] [threads] [outfile] [machine_key]
        & .\build\Release\matrix_profiler.exe 1000000000000 20 $outfile $machine
        
        $end = Get-Date
        Write-Host "Completed in $(($end - $start).TotalMinutes) minutes." -ForegroundColor Green
        Read-Host "Press Enter to return to menu..."
    }

} while ($true)
