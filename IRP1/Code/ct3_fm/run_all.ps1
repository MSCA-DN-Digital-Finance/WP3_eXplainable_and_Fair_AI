# Stop on PowerShell errors
$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Message,
        [string]$Env,
        [string]$Script
    )
    Write-Host $Message
    conda run -n $Env --no-capture-output python $Script
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Message (env=$Env, script=$Script, exitcode=$LASTEXITCODE)"
    }
}

Run-Step "Creating trajectories..."         "ct3-core"          "create_trajs.py"
Run-Step "Running Chronos predictions..."   "chronos"           "create_preds_chronos.py"
Run-Step "Running TimesFM predictions..."   "timesfm"           "create_preds_timesfm.py"
Run-Step "Computing CT3 metrics..."         "ct3-core"          "create_ct3_metrics.py"

Write-Host "Done."
