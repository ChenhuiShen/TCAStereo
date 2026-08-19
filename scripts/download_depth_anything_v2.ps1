param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("vits", "vitb", "vitl")]
    [string]$Encoder = "vitl",

    [Parameter(Mandatory = $false)]
    [string]$OutputDirectory = "checkpoints/depth_anything_v2"
)

$ErrorActionPreference = "Stop"

$modelNames = @{
    "vits" = "Small"
    "vitb" = "Base"
    "vitl" = "Large"
}

$modelName = $modelNames[$Encoder]
$fileName = "depth_anything_v2_$Encoder.pth"
$url = "https://huggingface.co/depth-anything/Depth-Anything-V2-$modelName/resolve/main/$fileName"
$destinationRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
$destination = Join-Path $destinationRoot $fileName

New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null
Write-Host "Downloading official Depth Anything V2 $modelName weights..."
Invoke-WebRequest -Uri $url -OutFile $destination
Write-Host "Saved: $destination"
