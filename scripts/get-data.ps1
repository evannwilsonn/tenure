# Downloads the IBM Telco Customer Churn CSV into data/ (it isn't committed to this repo).
$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path "$root/data" | Out-Null
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" -OutFile "$root/data/telco-customer-churn.csv"
Write-Host "Saved data/telco-customer-churn.csv"
