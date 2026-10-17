#!/usr/bin/env sh
# Downloads the IBM Telco Customer Churn CSV into data/ (it isn't committed to this repo).
set -e
cd "$(dirname "$0")/.."
mkdir -p data
curl -fsSL -o data/telco-customer-churn.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
echo "Saved data/telco-customer-churn.csv ($(wc -l < data/telco-customer-churn.csv) lines)"
