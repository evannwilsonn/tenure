using Tenure.Core;

namespace Tenure.Api;

/// <summary>
/// The /predict body. Everything is nullable so a missing field becomes a clear 400 instead of
/// a silent default. TotalCharges is optional: if it's left out, tenure × monthly charges is used.
/// </summary>
public sealed record CustomerRequest(
    string? Gender, bool? SeniorCitizen, string? Partner, string? Dependents, double? Tenure,
    string? PhoneService, string? MultipleLines, string? InternetService, string? OnlineSecurity,
    string? OnlineBackup, string? DeviceProtection, string? TechSupport, string? StreamingTV,
    string? StreamingMovies, string? Contract, string? PaperlessBilling, string? PaymentMethod,
    double? MonthlyCharges, double? TotalCharges)
{
    public (Customer? Customer, List<string> Errors) ToCustomer()
    {
        var missing = new List<string>();
        string S(string? v, string name) { if (string.IsNullOrWhiteSpace(v)) missing.Add(name); return v?.Trim() ?? ""; }
        double D(double? v, string name) { if (v is null) missing.Add(name); return v ?? 0; }
        if (SeniorCitizen is null) missing.Add("seniorCitizen");

        var c = new Customer
        {
            Gender = S(Gender, "gender"),
            Senior = SeniorCitizen ?? false,
            Partner = S(Partner, "partner"),
            Dependents = S(Dependents, "dependents"),
            Tenure = D(Tenure, "tenure"),
            PhoneService = S(PhoneService, "phoneService"),
            MultipleLines = S(MultipleLines, "multipleLines"),
            InternetService = S(InternetService, "internetService"),
            OnlineSecurity = S(OnlineSecurity, "onlineSecurity"),
            OnlineBackup = S(OnlineBackup, "onlineBackup"),
            DeviceProtection = S(DeviceProtection, "deviceProtection"),
            TechSupport = S(TechSupport, "techSupport"),
            StreamingTV = S(StreamingTV, "streamingTV"),
            StreamingMovies = S(StreamingMovies, "streamingMovies"),
            Contract = S(Contract, "contract"),
            PaperlessBilling = S(PaperlessBilling, "paperlessBilling"),
            PaymentMethod = S(PaymentMethod, "paymentMethod"),
            MonthlyCharges = D(MonthlyCharges, "monthlyCharges"),
        };
        c = c with { TotalCharges = TotalCharges ?? c.Tenure * c.MonthlyCharges };
        return missing.Count > 0
            ? (null, missing.Select(m => $"Missing field: {m}").ToList())
            : (c, new List<string>());
    }
}
