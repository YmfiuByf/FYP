$ErrorActionPreference = "Continue"

Set-Location "D:\pycharmProject\FYP"

$figRoot = "outputs\figures"
$srcFig = Join-Path $figRoot "Figure"
$checkpointDir = "outputs\checkpoints"

function Ensure-Dir {
    param([string]$Path)
    if (!(Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Move-IfExists {
    param(
        [string]$Source,
        [string]$Destination
    )
    if (Test-Path $Source) {
        Ensure-Dir $Destination
        Move-Item $Source $Destination -Force
        Write-Host "Moved: $Source -> $Destination"
    }
}

Ensure-Dir $checkpointDir
Ensure-Dir "$figRoot\default\GNB"
Ensure-Dir "$figRoot\default\IMB_HistGradientBoosting"
Ensure-Dir "$figRoot\default\KNN"
Ensure-Dir "$figRoot\default\SGD"
Ensure-Dir "$figRoot\default\SVM"
Ensure-Dir "$figRoot\default\Tree"

Ensure-Dir "$figRoot\SAVEE\GNB"
Ensure-Dir "$figRoot\SAVEE\IMB_HistGradientBoosting"
Ensure-Dir "$figRoot\SAVEE\KNN"
Ensure-Dir "$figRoot\SAVEE\SGD"
Ensure-Dir "$figRoot\SAVEE\SVM"
Ensure-Dir "$figRoot\SAVEE\Tree"

Move-IfExists "LSTM.dat" $checkpointDir
Move-IfExists "LSTM2.dat" $checkpointDir

if (Test-Path $srcFig) {
    Get-ChildItem $srcFig -File -Filter "*.png" | ForEach-Object {
        $name = $_.Name

        if ($name -like "SAVEE_GNB_*") {
            Move-Item $_.FullName "$figRoot\SAVEE\GNB\" -Force
        }
        elseif ($name -like "SAVEE_IMB_HistGradientBoosting_*") {
            Move-Item $_.FullName "$figRoot\SAVEE\IMB_HistGradientBoosting\" -Force
        }
        elseif ($name -like "SAVEE_KNN_*") {
            Move-Item $_.FullName "$figRoot\SAVEE\KNN\" -Force
        }
        elseif ($name -like "SAVEE_SGD_*") {
            Move-Item $_.FullName "$figRoot\SAVEE\SGD\" -Force
        }
        elseif ($name -like "SAVEE_SVM_*") {
            Move-Item $_.FullName "$figRoot\SAVEE\SVM\" -Force
        }
        elseif ($name -like "SAVEE_Tree_*") {
            Move-Item $_.FullName "$figRoot\SAVEE\Tree\" -Force
        }
        elseif ($name -like "GNB_*") {
            Move-Item $_.FullName "$figRoot\default\GNB\" -Force
        }
        elseif ($name -like "IMB_HistGradientBoosting_*") {
            Move-Item $_.FullName "$figRoot\default\IMB_HistGradientBoosting\" -Force
        }
        elseif ($name -like "KNN_*") {
            Move-Item $_.FullName "$figRoot\default\KNN\" -Force
        }
        elseif ($name -like "SGD_*") {
            Move-Item $_.FullName "$figRoot\default\SGD\" -Force
        }
        elseif ($name -like "SVM_*") {
            Move-Item $_.FullName "$figRoot\default\SVM\" -Force
        }
        elseif ($name -like "Tree_*") {
            Move-Item $_.FullName "$figRoot\default\Tree\" -Force
        }
    }
}

if (Test-Path "src\evaluation\eval_transformer+SVM" -PathType Leaf) {
    Remove-Item "src\evaluation\eval_transformer+SVM" -Force
    Write-Host "Removed: src\evaluation\eval_transformer+SVM"
}

Get-ChildItem . -Directory -Recurse |
Sort-Object FullName -Descending |
Where-Object { @(Get-ChildItem $_.FullName -Force).Count -eq 0 } |
ForEach-Object {
    Remove-Item $_.FullName -Force
    Write-Host "Removed empty dir: $($_.FullName)"
}

Write-Host ""
Write-Host "Done."