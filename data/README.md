# Data

The IBM Telco Customer Churn dataset: 7,043 customers of a fictional telecom company, 21 columns, one row per customer. `Churn` is whether they left in the last month (26.5% did).

Source: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv). That repository doesn't state a license, so the CSV isn't copied here. Download it with:

```sh
./scripts/get-data.sh        # macOS / Linux
./scripts/get-data.ps1       # Windows PowerShell
```

One quirk: 11 customers have a blank `TotalCharges`. All of them have `tenure` 0, meaning they joined this month and haven't been billed yet, so the loader reads the blank as 0 rather than dropping the rows.
