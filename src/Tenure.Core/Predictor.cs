namespace Tenure.Core;

/// <summary>One input's push on the prediction, in log-odds, relative to an average training customer.</summary>
public sealed record Reason(string Input, string Value, double Effect);

public sealed record Prediction(double Probability, bool Churn, double Threshold, IReadOnlyList<Reason> RaisesRisk, IReadOnlyList<Reason> LowersRisk);

/// <summary>
/// Scores one customer with the exported model and explains it. Because every input is
/// standardized against the training data, coef × standardized value is how far that input
/// moves this customer's log-odds away from the average customer's. Dummy columns and
/// engineered columns are summed back into the input a person would recognise.
/// </summary>
public sealed class Predictor
{
    private readonly ExportedModel _model;

    public Predictor(ExportedModel model) => _model = model;

    public double Threshold => _model.Threshold;

    public static Predictor Load(string path) =>
        new(System.Text.Json.JsonSerializer.Deserialize<ExportedModel>(File.ReadAllText(path), ExportedModel.Json)
            ?? throw new InvalidDataException($"Could not read model from {path}"));

    /// <summary>The allowed answers for each categorical input (from the dataset).</summary>
    public static readonly IReadOnlyDictionary<string, string[]> Allowed = new Dictionary<string, string[]>
    {
        [nameof(Customer.Gender)] = new[] { "Female", "Male" },
        [nameof(Customer.Partner)] = new[] { "No", "Yes" },
        [nameof(Customer.Dependents)] = new[] { "No", "Yes" },
        [nameof(Customer.PhoneService)] = new[] { "No", "Yes" },
        [nameof(Customer.MultipleLines)] = new[] { "No", "No phone service", "Yes" },
        [nameof(Customer.InternetService)] = new[] { "DSL", "Fiber optic", "No" },
        [nameof(Customer.OnlineSecurity)] = new[] { "No", "No internet service", "Yes" },
        [nameof(Customer.OnlineBackup)] = new[] { "No", "No internet service", "Yes" },
        [nameof(Customer.DeviceProtection)] = new[] { "No", "No internet service", "Yes" },
        [nameof(Customer.TechSupport)] = new[] { "No", "No internet service", "Yes" },
        [nameof(Customer.StreamingTV)] = new[] { "No", "No internet service", "Yes" },
        [nameof(Customer.StreamingMovies)] = new[] { "No", "No internet service", "Yes" },
        [nameof(Customer.Contract)] = new[] { "Month-to-month", "One year", "Two year" },
        [nameof(Customer.PaperlessBilling)] = new[] { "No", "Yes" },
        [nameof(Customer.PaymentMethod)] = new[] { "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check" },
    };

    /// <summary>Problems with a customer's values, or an empty list if it can be scored.</summary>
    public static List<string> Validate(Customer c)
    {
        var errors = new List<string>();
        foreach (var (col, allowed) in Allowed)
        {
            var v = c.Category(col);
            if (!allowed.Contains(v)) errors.Add($"{col} must be one of: {string.Join(", ", allowed)} (got \"{v}\")");
        }
        if (c.Tenure < 0 || c.Tenure > 120) errors.Add("Tenure must be between 0 and 120 months");
        if (c.MonthlyCharges < 0 || c.MonthlyCharges > 500) errors.Add("MonthlyCharges must be between 0 and 500");
        if (c.TotalCharges < 0) errors.Add("TotalCharges can't be negative");
        if (c.PhoneService == "No" && c.MultipleLines != "No phone service") errors.Add("MultipleLines must be \"No phone service\" when PhoneService is \"No\"");
        if (c.InternetService == "No")
            foreach (var col in new[] { "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies" })
                if (c.Category(col) != "No internet service") errors.Add($"{col} must be \"No internet service\" when InternetService is \"No\"");
        return errors;
    }

    /// <summary>The input a person would recognise for a model column.</summary>
    public static string SourceInput(string column) => column switch
    {
        "LogTenure" or "NewCustomer" => nameof(Customer.Tenure),
        "ServiceCount" => "Add-on services",
        _ => column,
    };

    private static string Display(Customer c, string input) => input switch
    {
        nameof(Customer.Tenure) => $"{c.Tenure:0} months",
        nameof(Customer.MonthlyCharges) => $"${c.MonthlyCharges:0.00}",
        nameof(Customer.TotalCharges) => $"${c.TotalCharges:0.00}",
        "Add-on services" => Encoder.ServiceCount(c).ToString(),
        _ => c.Category(input),
    };

    public Prediction Predict(Customer c, int topReasons = 3)
    {
        double z = _model.Intercept;
        var effect = new Dictionary<string, double>();
        foreach (var f in _model.Features)
        {
            double e = f.Coef * (Encoder.RawValue(c, f.Column, f.Level) - f.Mean) / f.Sd;
            z += e;
            var key = SourceInput(f.Column);
            effect[key] = effect.GetValueOrDefault(key) + e;
        }
        double p = Logistic.Sigmoid(z);
        var reasons = effect.Select(kv => new Reason(kv.Key, Display(c, kv.Key), Math.Round(kv.Value, 4))).ToList();
        return new Prediction(
            Math.Round(p, 4), p >= _model.Threshold, Math.Round(_model.Threshold, 4),
            reasons.Where(r => r.Effect > 0.01).OrderByDescending(r => r.Effect).Take(topReasons).ToList(),
            reasons.Where(r => r.Effect < -0.01).OrderBy(r => r.Effect).Take(topReasons).ToList());
    }
}
