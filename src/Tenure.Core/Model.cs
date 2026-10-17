using System.Text.Json;
using System.Text.Json.Serialization;

namespace Tenure.Core;

/// <summary>One model input: a numeric column, or one level of a categorical column (0/1).</summary>
public sealed record FeatureSpec(string Name, string Column, string? Level, double Mean, double Sd);

/// <summary>
/// Turns customers into standardized feature vectors. Fit on training rows only: category
/// levels, means and standard deviations all come from train, then get reused on test.
/// </summary>
public sealed class Encoder
{
    public static readonly string[] Categorical =
    {
        nameof(Customer.Senior), nameof(Customer.Gender), nameof(Customer.Partner), nameof(Customer.Dependents),
        nameof(Customer.PhoneService), nameof(Customer.MultipleLines), nameof(Customer.InternetService),
        nameof(Customer.OnlineSecurity), nameof(Customer.OnlineBackup), nameof(Customer.DeviceProtection),
        nameof(Customer.TechSupport), nameof(Customer.StreamingTV), nameof(Customer.StreamingMovies),
        nameof(Customer.Contract), nameof(Customer.PaperlessBilling), nameof(Customer.PaymentMethod),
    };
    public static readonly string[] Numeric = { nameof(Customer.Tenure), nameof(Customer.MonthlyCharges), nameof(Customer.TotalCharges) };

    /// <summary>Extra columns tried in the improvement step (see Numeric for the base set).</summary>
    public static readonly string[] Engineered = { "LogTenure", "ServiceCount", "NewCustomer" };

    public IReadOnlyList<FeatureSpec> Features { get; }

    private Encoder(List<FeatureSpec> features) => Features = features;

    public static Encoder Fit(IReadOnlyList<Customer> train, bool engineered = false, ISet<string>? drop = null)
    {
        var raw = new List<(string Name, string Column, string? Level)>();
        foreach (var col in Numeric.Concat(engineered ? Engineered : Array.Empty<string>()))
            if (drop is null || !drop.Contains(col)) raw.Add((col, col, null));
        foreach (var col in Categorical)
        {
            if (drop is not null && drop.Contains(col)) continue;
            // The first level alphabetically is the reference category and gets no column.
            var levels = train.Select(r => r.Category(col)).Distinct().OrderBy(l => l, StringComparer.Ordinal).Skip(1);
            foreach (var level in levels) raw.Add(($"{col}={level}", col, level));
        }
        var specs = new List<FeatureSpec>();
        foreach (var (name, column, level) in raw)
        {
            var xs = train.Select(r => RawValue(r, column, level)).ToArray();
            double mean = xs.Average();
            double sd = Math.Sqrt(xs.Sum(x => (x - mean) * (x - mean)) / (xs.Length - 1));
            if (sd < 1e-12) continue; // constant in train: carries no information
            specs.Add(new FeatureSpec(name, column, level, mean, sd));
        }
        return new Encoder(specs);
    }

    public static Encoder FromSpecs(IEnumerable<FeatureSpec> specs) => new(specs.ToList());

    public static double RawValue(Customer r, string column, string? level)
    {
        if (level is not null) return r.Category(column) == level ? 1 : 0;
        return column switch
        {
            nameof(Customer.Tenure) => r.Tenure,
            nameof(Customer.MonthlyCharges) => r.MonthlyCharges,
            nameof(Customer.TotalCharges) => r.TotalCharges,
            "LogTenure" => Math.Log(1 + r.Tenure),
            "ServiceCount" => ServiceCount(r),
            "NewCustomer" => r.Tenure <= 6 ? 1 : 0,
            _ => throw new ArgumentException($"Unknown numeric column {column}")
        };
    }

    /// <summary>How many optional services the customer pays for (phone lines, internet add-ons, streaming).</summary>
    public static int ServiceCount(Customer r) =>
        new[] { r.MultipleLines, r.OnlineSecurity, r.OnlineBackup, r.DeviceProtection, r.TechSupport, r.StreamingTV, r.StreamingMovies }
            .Count(v => v == "Yes");

    public double[] Encode(Customer r)
    {
        var x = new double[Features.Count];
        for (int j = 0; j < x.Length; j++)
        {
            var f = Features[j];
            x[j] = (RawValue(r, f.Column, f.Level) - f.Mean) / f.Sd;
        }
        return x;
    }

    public double[][] Encode(IEnumerable<Customer> rows) => rows.Select(Encode).ToArray();
}

/// <summary>
/// Logistic regression fitted from scratch by Newton's method (iteratively reweighted least
/// squares) with an L2 penalty on the weights, not the intercept. Optional per-row weights
/// are how class weighting is done.
/// </summary>
public sealed class Logistic
{
    public double Intercept { get; private set; }
    public double[] Weights { get; private set; } = Array.Empty<double>();
    public int Iterations { get; private set; }

    public static double Sigmoid(double z) => z >= 0 ? 1 / (1 + Math.Exp(-z)) : Math.Exp(z) / (1 + Math.Exp(z));

    public static Logistic Fit(double[][] x, bool[] y, double l2 = 1.0, double[]? rowWeight = null, int maxIter = 50, double tol = 1e-10)
    {
        int n = x.Length, p = x[0].Length, d = p + 1; // index 0 is the intercept
        var beta = new double[d];
        var model = new Logistic();
        for (int it = 1; it <= maxIter; it++)
        {
            var grad = new double[d];
            var hess = new double[d, d];
            for (int i = 0; i < n; i++)
            {
                double z = beta[0];
                for (int j = 0; j < p; j++) z += beta[j + 1] * x[i][j];
                double mu = Sigmoid(z), w = rowWeight?[i] ?? 1.0;
                double r = w * ((y[i] ? 1 : 0) - mu), s = w * mu * (1 - mu);
                grad[0] += r;
                hess[0, 0] += s;
                for (int j = 0; j < p; j++)
                {
                    double xij = x[i][j];
                    grad[j + 1] += r * xij;
                    hess[0, j + 1] += s * xij;
                    for (int k = j; k < p; k++) hess[j + 1, k + 1] += s * xij * x[i][k];
                }
            }
            for (int j = 1; j < d; j++)
            {
                grad[j] -= l2 * beta[j];
                hess[j, j] += l2;
            }
            for (int j = 0; j < d; j++) for (int k = 0; k < j; k++) hess[j, k] = hess[k, j];
            var step = Linear.SolveSpd(hess, grad);
            double change = 0;
            for (int j = 0; j < d; j++) { beta[j] += step[j]; change = Math.Max(change, Math.Abs(step[j])); }
            model.Iterations = it;
            if (change < tol) break;
        }
        model.Intercept = beta[0];
        model.Weights = beta[1..];
        return model;
    }

    public static Logistic FromWeights(double intercept, double[] weights) => new() { Intercept = intercept, Weights = weights };

    public double Score(double[] x)
    {
        double z = Intercept;
        for (int j = 0; j < Weights.Length; j++) z += Weights[j] * x[j];
        return z;
    }

    public double Probability(double[] x) => Sigmoid(Score(x));
}

public static class Linear
{
    /// <summary>Solves A x = b for a symmetric positive definite A by Cholesky decomposition.</summary>
    public static double[] SolveSpd(double[,] a, double[] b)
    {
        int n = b.Length;
        var l = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
            {
                double sum = a[i, j];
                for (int k = 0; k < j; k++) sum -= l[i, k] * l[j, k];
                if (i == j)
                {
                    if (sum <= 0) throw new InvalidOperationException("Matrix is not positive definite; raise the L2 penalty.");
                    l[i, i] = Math.Sqrt(sum);
                }
                else l[i, j] = sum / l[j, j];
            }
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = b[i];
            for (int k = 0; k < i; k++) sum -= l[i, k] * y[k];
            y[i] = sum / l[i, i];
        }
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = y[i];
            for (int k = i + 1; k < n; k++) sum -= l[k, i] * x[k];
            x[i] = sum / l[i, i];
        }
        return x;
    }
}

/// <summary>The trained model as plain numbers: what the API and the browser demo load.</summary>
public sealed record ExportedModel
{
    public required double Intercept { get; init; }
    public required double Threshold { get; init; }
    public required List<ExportedFeature> Features { get; init; }
    public Dictionary<string, object>? Meta { get; init; }

    public static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull };

    public static ExportedModel From(Encoder enc, Logistic m, double threshold, Dictionary<string, object>? meta = null) => new()
    {
        Intercept = m.Intercept,
        Threshold = threshold,
        Features = enc.Features.Select((f, j) => new ExportedFeature(f.Name, f.Column, f.Level, f.Mean, f.Sd, m.Weights[j])).ToList(),
        Meta = meta,
    };

    public (Encoder Encoder, Logistic Model) Load() =>
        (Encoder.FromSpecs(Features.Select(f => new FeatureSpec(f.Name, f.Column, f.Level, f.Mean, f.Sd))),
         Logistic.FromWeights(Intercept, Features.Select(f => f.Coef).ToArray()));
}

public sealed record ExportedFeature(string Name, string Column, string? Level, double Mean, double Sd, double Coef);
