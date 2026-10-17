using Tenure.Core;

namespace Tenure.Tests;

public class DataTests
{
    private const string Header = "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn";

    [Fact]
    public void Loader_reads_blank_total_charges_as_zero()
    {
        var path = Path.GetTempFileName();
        File.WriteAllLines(path, new[]
        {
            Header,
            "A-1,Female,0,Yes,No,0,Yes,No,DSL,No,No,No,No,No,No,Two year,Yes,Mailed check,52.55, ,No",
            "A-2,Male,1,No,No,12,Yes,Yes,Fiber optic,No,No,No,No,Yes,No,Month-to-month,Yes,Electronic check,90.10,1081.2,Yes",
        });
        var rows = TelcoData.Load(path);
        Assert.Equal(2, rows.Count);
        Assert.Equal(0, rows[0].TotalCharges);
        Assert.False(rows[0].Churn);
        Assert.True(rows[1].Senior);
        Assert.Equal("Yes", rows[1].Category("Senior"));
        Assert.Equal(1081.2, rows[1].TotalCharges, 6);
        Assert.True(rows[1].Churn);
    }

    private static List<Customer> Synthetic(int n, double churnShare)
    {
        int churners = (int)(n * churnShare);
        return Enumerable.Range(0, n).Select(i => new Customer { Id = $"c{i}", Churn = i < churners }).ToList();
    }

    [Fact]
    public void Stratified_split_keeps_the_churn_rate_and_loses_no_rows()
    {
        var rows = Synthetic(1000, 0.265);
        var (train, test) = TelcoData.StratifiedSplit(rows, 0.2, 7);
        Assert.Equal(1000, train.Count + test.Count);
        Assert.Empty(train.Select(r => r.Id).Intersect(test.Select(r => r.Id)));
        Assert.Equal(200, test.Count);
        Assert.Equal(0.265, test.Average(r => r.Churn ? 1.0 : 0), 2);
        Assert.Equal(0.265, train.Average(r => r.Churn ? 1.0 : 0), 2);
    }

    [Fact]
    public void Stratified_folds_are_balanced()
    {
        var rows = Synthetic(1000, 0.265);
        var fold = TelcoData.StratifiedFolds(rows, 5, 7);
        for (int k = 0; k < 5; k++)
        {
            var idx = Enumerable.Range(0, rows.Count).Where(i => fold[i] == k).ToList();
            Assert.Equal(200, idx.Count);
            Assert.InRange(idx.Count(i => rows[i].Churn), 52, 54);
        }
    }
}

public class LogisticTests
{
    [Fact]
    public void Matches_the_closed_form_for_one_binary_input()
    {
        // With one 0/1 input and (almost) no penalty, the fit has a known answer:
        // intercept = logit(churn rate when x = 0), coef = logit(rate | x = 1) − logit(rate | x = 0).
        var x = new List<double[]>();
        var y = new List<bool>();
        void Add(double v, bool label, int count) { for (int i = 0; i < count; i++) { x.Add(new[] { v }); y.Add(label); } }
        Add(0, true, 20); Add(0, false, 80); // 20% churn
        Add(1, true, 60); Add(1, false, 40); // 60% churn
        var m = Logistic.Fit(x.ToArray(), y.ToArray(), l2: 1e-9);
        static double Logit(double p) => Math.Log(p / (1 - p));
        Assert.Equal(Logit(0.2), m.Intercept, 6);
        Assert.Equal(Logit(0.6) - Logit(0.2), m.Weights[0], 6);
    }

    [Fact]
    public void Gradient_is_zero_at_the_penalized_solution()
    {
        var rng = new Random(3);
        var x = Enumerable.Range(0, 400).Select(_ => new[] { rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1, rng.NextDouble() }).ToArray();
        var y = x.Select(r => rng.NextDouble() < Logistic.Sigmoid(0.5 + 2 * r[0] - r[1])).ToArray();
        const double l2 = 3;
        var m = Logistic.Fit(x, y, l2);
        var grad = new double[4];
        for (int i = 0; i < x.Length; i++)
        {
            double r = (y[i] ? 1 : 0) - m.Probability(x[i]);
            grad[0] += r;
            for (int j = 0; j < 3; j++) grad[j + 1] += r * x[i][j];
        }
        for (int j = 0; j < 3; j++) grad[j + 1] -= l2 * m.Weights[j];
        Assert.All(grad, g => Assert.True(Math.Abs(g) < 1e-8, $"gradient {g}"));
    }

    [Fact]
    public void Row_weights_act_like_duplicated_rows()
    {
        var x = new[] { new[] { 0.0 }, new[] { 1.0 }, new[] { 1.0 }, new[] { 0.0 } };
        var y = new[] { true, true, false, false };
        var weighted = Logistic.Fit(x, y, 0.5, new[] { 3.0, 1, 1, 1 });
        var duplicated = Logistic.Fit(new[] { x[0], x[0], x[0], x[1], x[2], x[3] }, new[] { true, true, true, true, false, false }, 0.5);
        Assert.Equal(duplicated.Intercept, weighted.Intercept, 9);
        Assert.Equal(duplicated.Weights[0], weighted.Weights[0], 9);
    }

    [Fact]
    public void Sigmoid_is_stable_at_extremes()
    {
        Assert.Equal(1.0, Logistic.Sigmoid(800), 12);
        Assert.Equal(0.0, Logistic.Sigmoid(-800), 12);
        Assert.Equal(0.5, Logistic.Sigmoid(0), 12);
    }
}

public class MetricsTests
{
    private static readonly bool[] Y = { true, false, true, false, false };
    private static readonly double[] P = { 0.9, 0.8, 0.6, 0.3, 0.1 };

    [Fact]
    public void Evaluate_counts_the_confusion_matrix()
    {
        var s = Metrics.Evaluate(P, Y, 0.5);
        Assert.Equal((2, 1, 0, 2), (s.TruePos, s.FalsePos, s.FalseNeg, s.TrueNeg));
        Assert.Equal(2.0 / 3, s.Precision, 9);
        Assert.Equal(1.0, s.Recall, 9);
        Assert.Equal(0.8, s.F1, 9);
    }

    [Fact]
    public void Roc_auc_counts_correctly_ordered_pairs()
    {
        // Pairs (churner, stayer): 0.9 beats all 3 stayers; 0.6 beats 0.3 and 0.1 but not 0.8, so 5 of 6.
        Assert.Equal(5.0 / 6, Metrics.RocAuc(P, Y), 9);
        Assert.Equal(0.5, Metrics.RocAuc(new[] { 0.4, 0.4, 0.4, 0.4 }, new[] { true, false, true, false }), 9);
    }

    [Fact]
    public void Average_precision_of_a_constant_score_is_the_base_rate()
    {
        var y = Enumerable.Range(0, 100).Select(i => i % 4 == 0).ToArray();
        Assert.Equal(0.25, Metrics.AveragePrecision(Enumerable.Repeat(0.3, 100).ToArray(), y), 9);
        // Hand-checked: hits at ranks 1 and 3, so (1/1 + 2/3) / 2.
        Assert.Equal((1 + 2.0 / 3) / 2, Metrics.AveragePrecision(P, Y), 9);
    }

    [Fact]
    public void Best_f1_threshold_is_found()
    {
        // Cutting at 0.6 flags {0.9, 0.8, 0.6}: P 2/3, R 1, F1 0.8, the best available.
        Assert.Equal(0.6, Metrics.BestF1Threshold(P, Y), 9);
    }
}

public class ExportTests
{
    private static Customer Sample(string contract, double tenure, bool churn) => new()
    {
        Gender = "Female", Partner = "No", Dependents = "No", Tenure = tenure, PhoneService = "Yes", MultipleLines = "No",
        InternetService = "DSL", OnlineSecurity = "No", OnlineBackup = "Yes", DeviceProtection = "No", TechSupport = "No",
        StreamingTV = "No", StreamingMovies = "No", Contract = contract, PaperlessBilling = "Yes", PaymentMethod = "Mailed check",
        MonthlyCharges = 50 + tenure, TotalCharges = tenure * (50 + tenure), Churn = churn,
    };

    private static List<Customer> Rows() =>
        Enumerable.Range(0, 60).Select(i => Sample(i % 3 == 0 ? "Month-to-month" : i % 3 == 1 ? "One year" : "Two year", i, i % 3 == 0 && i % 2 == 0)).ToList();

    [Fact]
    public void Exported_model_round_trips_through_json()
    {
        var rows = Rows();
        var enc = Encoder.Fit(rows, engineered: true);
        var m = Logistic.Fit(enc.Encode(rows), rows.Select(r => r.Churn).ToArray());
        var json = System.Text.Json.JsonSerializer.Serialize(ExportedModel.From(enc, m, 0.3), ExportedModel.Json);
        var (enc2, m2) = System.Text.Json.JsonSerializer.Deserialize<ExportedModel>(json, ExportedModel.Json)!.Load();
        foreach (var r in rows) Assert.Equal(m.Probability(enc.Encode(r)), m2.Probability(enc2.Encode(r)), 12);
    }

    [Fact]
    public void Encoder_drops_constant_columns_and_the_reference_level()
    {
        var enc = Encoder.Fit(Rows());
        var names = enc.Features.Select(f => f.Name).ToList();
        Assert.DoesNotContain("Gender=Male", names);              // everyone is Female: no column
        Assert.DoesNotContain("Contract=Month-to-month", names);  // reference level
        Assert.Contains("Contract=One year", names);
        Assert.Contains("Contract=Two year", names);
    }

    [Fact]
    public void Predictor_matches_the_model_and_groups_engineered_columns()
    {
        var rows = Rows();
        var enc = Encoder.Fit(rows, engineered: true);
        var m = Logistic.Fit(enc.Encode(rows), rows.Select(r => r.Churn).ToArray());
        var predictor = new Predictor(ExportedModel.From(enc, m, 0.3));
        var c = rows[3];
        var pred = predictor.Predict(c, topReasons: 100);
        Assert.Equal(m.Probability(enc.Encode(c)), pred.Probability, 3);
        Assert.Equal(pred.Probability >= 0.3, pred.Churn);
        Assert.DoesNotContain(pred.RaisesRisk.Concat(pred.LowersRisk), r => r.Input is "LogTenure" or "NewCustomer" or "ServiceCount");
    }

    [Fact]
    public void Validation_catches_impossible_combinations()
    {
        var c = Sample("Weekly", 5, false) with { PhoneService = "No" };
        var errors = Predictor.Validate(c);
        Assert.Contains(errors, e => e.StartsWith("Contract"));
        Assert.Contains(errors, e => e.StartsWith("MultipleLines"));
        Assert.Empty(Predictor.Validate(Sample("One year", 5, false)));
    }
}
