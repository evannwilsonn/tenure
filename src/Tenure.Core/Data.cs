using System.Globalization;

namespace Tenure.Core;

/// <summary>One row of the IBM Telco Customer Churn data, as the CSV gives it.</summary>
public sealed record Customer
{
    public string Id { get; init; } = "";
    public string Gender { get; init; } = "";
    public bool Senior { get; init; }
    public string Partner { get; init; } = "";
    public string Dependents { get; init; } = "";
    public double Tenure { get; init; }
    public string PhoneService { get; init; } = "";
    public string MultipleLines { get; init; } = "";
    public string InternetService { get; init; } = "";
    public string OnlineSecurity { get; init; } = "";
    public string OnlineBackup { get; init; } = "";
    public string DeviceProtection { get; init; } = "";
    public string TechSupport { get; init; } = "";
    public string StreamingTV { get; init; } = "";
    public string StreamingMovies { get; init; } = "";
    public string Contract { get; init; } = "";
    public string PaperlessBilling { get; init; } = "";
    public string PaymentMethod { get; init; } = "";
    public double MonthlyCharges { get; init; }
    public double TotalCharges { get; init; }
    public bool Churn { get; init; }

    /// <summary>Categorical answers by column name, so the encoder can treat them uniformly.</summary>
    public string Category(string column) => column switch
    {
        nameof(Gender) => Gender,
        nameof(Partner) => Partner,
        nameof(Dependents) => Dependents,
        nameof(PhoneService) => PhoneService,
        nameof(MultipleLines) => MultipleLines,
        nameof(InternetService) => InternetService,
        nameof(OnlineSecurity) => OnlineSecurity,
        nameof(OnlineBackup) => OnlineBackup,
        nameof(DeviceProtection) => DeviceProtection,
        nameof(TechSupport) => TechSupport,
        nameof(StreamingTV) => StreamingTV,
        nameof(StreamingMovies) => StreamingMovies,
        nameof(Contract) => Contract,
        nameof(PaperlessBilling) => PaperlessBilling,
        nameof(PaymentMethod) => PaymentMethod,
        nameof(Senior) => Senior ? "Yes" : "No",
        _ => throw new ArgumentException($"Unknown categorical column {column}")
    };
}

public static class TelcoData
{
    /// <summary>
    /// Reads the CSV. The 11 customers with tenure 0 have a blank TotalCharges: they joined
    /// this month and haven't been billed yet, so their total is 0, not missing.
    /// </summary>
    public static List<Customer> Load(string path)
    {
        var lines = File.ReadAllLines(path);
        var header = lines[0].Split(',');
        var c = new Dictionary<string, int>();
        for (int i = 0; i < header.Length; i++) c[header[i].Trim()] = i;
        foreach (var need in new[] { "customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn", "Contract" })
            if (!c.ContainsKey(need)) throw new InvalidDataException($"Missing column {need}");
        var inv = CultureInfo.InvariantCulture;
        var rows = new List<Customer>(lines.Length - 1);
        foreach (var line in lines.Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var f = line.Split(',');
            var total = f[c["TotalCharges"]].Trim();
            rows.Add(new Customer
            {
                Id = f[c["customerID"]],
                Gender = f[c["gender"]],
                Senior = f[c["SeniorCitizen"]] == "1",
                Partner = f[c["Partner"]],
                Dependents = f[c["Dependents"]],
                Tenure = double.Parse(f[c["tenure"]], inv),
                PhoneService = f[c["PhoneService"]],
                MultipleLines = f[c["MultipleLines"]],
                InternetService = f[c["InternetService"]],
                OnlineSecurity = f[c["OnlineSecurity"]],
                OnlineBackup = f[c["OnlineBackup"]],
                DeviceProtection = f[c["DeviceProtection"]],
                TechSupport = f[c["TechSupport"]],
                StreamingTV = f[c["StreamingTV"]],
                StreamingMovies = f[c["StreamingMovies"]],
                Contract = f[c["Contract"]],
                PaperlessBilling = f[c["PaperlessBilling"]],
                PaymentMethod = f[c["PaymentMethod"]],
                MonthlyCharges = double.Parse(f[c["MonthlyCharges"]], inv),
                TotalCharges = total.Length == 0 ? 0 : double.Parse(total, inv),
                Churn = f[c["Churn"]].Trim() == "Yes",
            });
        }
        return rows;
    }

    /// <summary>
    /// Stratified split: the same churn rate in both halves, so a 26.5% minority class
    /// doesn't drift between train and test by chance.
    /// </summary>
    public static (List<Customer> Train, List<Customer> Test) StratifiedSplit(IReadOnlyList<Customer> rows, double testShare, int seed)
    {
        var rng = new Random(seed);
        var train = new List<Customer>();
        var test = new List<Customer>();
        foreach (var group in rows.GroupBy(r => r.Churn).OrderBy(g => g.Key))
        {
            var shuffled = group.OrderBy(_ => rng.Next()).ToList();
            int nTest = (int)Math.Round(shuffled.Count * testShare);
            test.AddRange(shuffled.Take(nTest));
            train.AddRange(shuffled.Skip(nTest));
        }
        return (train, test);
    }

    /// <summary>Stratified k folds: a fold index for every row, each fold with the same churn rate.</summary>
    public static int[] StratifiedFolds(IReadOnlyList<Customer> rows, int k, int seed)
    {
        var rng = new Random(seed);
        var fold = new int[rows.Count];
        foreach (var group in Enumerable.Range(0, rows.Count).GroupBy(i => rows[i].Churn).OrderBy(g => g.Key))
        {
            var idx = group.OrderBy(_ => rng.Next()).ToList();
            for (int j = 0; j < idx.Count; j++) fold[idx[j]] = j % k;
        }
        return fold;
    }
}
