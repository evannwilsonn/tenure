// Tenure: training and evaluation.
//
// Runs the whole loop in order: split the data, score two baselines, fit the hand-written
// logistic regression and check it against ML.NET, try gradient-boosted trees, run the
// improvement experiments with 5-fold cross-validation on the training set only, then
// open the test set exactly once for the final numbers. Writes results/*.json and the
// exported model that the API and the browser demo load.
//
//   dotnet run --project src/Tenure.Train -c Release

using System.Text.Json;
using Tenure.Core;

const int Seed = 2026;
const int Folds = 5;

string root = FindRoot();
string dataPath = Path.Combine(root, "data", "telco-customer-churn.csv");
if (!File.Exists(dataPath))
{
    Console.Error.WriteLine($"Missing {dataPath}. Run scripts/get-data.sh first.");
    return 1;
}
string resultsDir = Path.Combine(root, "results");
Directory.CreateDirectory(resultsDir);

// ---- Data -------------------------------------------------------------------------------
var all = TelcoData.Load(dataPath);
var (train, test) = TelcoData.StratifiedSplit(all, 0.2, Seed);
var fold = TelcoData.StratifiedFolds(train, Folds, Seed);
bool[] yTrain = train.Select(r => r.Churn).ToArray();
bool[] yTest = test.Select(r => r.Churn).ToArray();
double churnRate = yTrain.Average(v => v ? 1.0 : 0.0);

Console.WriteLine($"Rows: {all.Count}  train {train.Count}  test {test.Count}");
Console.WriteLine($"Churn rate: all {all.Average(r => r.Churn ? 1.0 : 0):P2}  train {churnRate:P2}  test {yTest.Average(v => v ? 1.0 : 0):P2}");

// A "recipe" turns training rows into a scoring function. Cross-validation calls it once per fold.
// Out-of-fold probabilities: each training row is scored by a model that never saw it.
double[] OutOfFold(Func<List<Customer>, Func<List<Customer>, double[]>> recipe)
{
    var p = new double[train.Count];
    for (int k = 0; k < Folds; k++)
    {
        var fitRows = train.Where((_, i) => fold[i] != k).ToList();
        var holdIdx = Enumerable.Range(0, train.Count).Where(i => fold[i] == k).ToArray();
        var score = recipe(fitRows);
        var probs = score(holdIdx.Select(i => train[i]).ToList());
        for (int j = 0; j < holdIdx.Length; j++) p[holdIdx[j]] = probs[j];
    }
    return p;
}

// Mean and spread of F1 across folds, so a difference can be compared with the fold-to-fold noise.
(double Mean, double Sd) FoldF1(double[] oof, double threshold)
{
    var f1s = Enumerable.Range(0, Folds).Select(k =>
    {
        var idx = Enumerable.Range(0, train.Count).Where(i => fold[i] == k).ToArray();
        return Metrics.Evaluate(idx.Select(i => oof[i]).ToArray(), idx.Select(i => yTrain[i]).ToArray(), threshold).F1;
    }).ToArray();
    double m = f1s.Average();
    return (m, Math.Sqrt(f1s.Sum(f => (f - m) * (f - m)) / (f1s.Length - 1)));
}

var cv = new List<Dictionary<string, object>>();
void Report(string name, string step, double[] oof, double threshold, string note)
{
    var s = Metrics.Evaluate(oof, yTrain, threshold);
    var (f1Mean, f1Sd) = FoldF1(oof, threshold);
    cv.Add(new()
    {
        ["name"] = name, ["step"] = step, ["note"] = note, ["threshold"] = R(threshold),
        ["precision"] = R(s.Precision), ["recall"] = R(s.Recall), ["f1"] = R(s.F1),
        ["f1FoldMean"] = R(f1Mean), ["f1FoldSd"] = R(f1Sd),
        ["rocAuc"] = R(s.RocAuc), ["prAuc"] = R(s.PrAuc), ["logLoss"] = R(s.LogLoss), ["accuracy"] = R(s.Accuracy),
    });
    Console.WriteLine($"  {name,-42} t={threshold:0.000}  P {s.Precision:0.000}  R {s.Recall:0.000}  F1 {s.F1:0.000} (±{f1Sd:0.000})  ROC {s.RocAuc:0.000}  PR {s.PrAuc:0.000}  acc {s.Accuracy:0.000}");
}

// ---- Baselines --------------------------------------------------------------------------
Console.WriteLine("\nBaselines (5-fold CV on train)");
// Majority class: nobody churns. 73% accurate, catches no one: why accuracy is the wrong metric here.
Report("Majority class (nobody churns)", "baseline",
    OutOfFold(rows => { double r = rows.Average(c => c.Churn ? 1.0 : 0); return xs => xs.Select(_ => r).ToArray(); }), 0.5,
    "Predicts the training churn rate for everyone, so it never flags anyone.");
// One-line rule a retention team might already use.
Report("Rule: month-to-month contract", "baseline",
    OutOfFold(_ => xs => xs.Select(c => c.Contract == "Month-to-month" ? 1.0 : 0.0).ToArray()), 0.5,
    "Flags every month-to-month customer. No training involved.");

// ---- Model ------------------------------------------------------------------------------
Func<List<Customer>, Func<List<Customer>, double[]>> LogisticRecipe(double l2, bool weighted = false, bool engineered = false) => rows =>
{
    var enc = Encoder.Fit(rows, engineered);
    var x = enc.Encode(rows);
    var y = rows.Select(r => r.Churn).ToArray();
    double[]? w = null;
    if (weighted)
    {
        double pos = y.Count(v => v), neg = y.Length - pos;
        w = y.Select(v => v ? neg / pos : 1.0).ToArray();
    }
    var m = Logistic.Fit(x, y, l2, w);
    return xs => enc.Encode(xs).Select(m.Probability).ToArray();
};

// Pick the L2 penalty by CV log loss (a probability-quality score, independent of any threshold).
Console.WriteLine("\nL2 penalty search (logistic regression, CV log loss)");
double bestL2 = 1, bestLL = double.MaxValue;
var l2Search = new List<Dictionary<string, object>>();
foreach (var l2 in new[] { 0.01, 0.1, 1, 10, 30, 100 })
{
    var oof = OutOfFold(LogisticRecipe(l2));
    double ll = Metrics.Evaluate(oof, yTrain, 0.5).LogLoss;
    l2Search.Add(new() { ["l2"] = l2, ["logLoss"] = R(ll), ["prAuc"] = R(Metrics.AveragePrecision(oof, yTrain)) });
    Console.WriteLine($"  l2={l2,-6} log loss {ll:0.00000}");
    if (ll < bestLL - 1e-6) { bestLL = ll; bestL2 = l2; }
}
Console.WriteLine($"  chosen l2={bestL2}");

Console.WriteLine("\nModels (5-fold CV on train, threshold 0.5)");
var oofLogit = OutOfFold(LogisticRecipe(bestL2));
Report("Logistic regression (from scratch)", "model", oofLogit, 0.5, $"Newton's method, L2 = {bestL2}, 0.5 threshold.");

var oofTree = OutOfFold(rows =>
{
    var enc = Encoder.Fit(rows);
    var predict = MlNet.FastTree(enc.Encode(rows), rows.Select(r => r.Churn).ToArray());
    return xs => predict(enc.Encode(xs));
});
Report("Gradient-boosted trees (ML.NET FastTree)", "model", oofTree, 0.5, "200 trees, 16 leaves, learning rate 0.05.");

// Check the hand-written solver against ML.NET's L-BFGS on the same features and penalty.
Console.WriteLine("\nCheck: hand-written Newton vs ML.NET L-BFGS (full training set)");
var encFull = Encoder.Fit(train);
var xFull = encFull.Encode(train);
var mine = Logistic.Fit(xFull, yTrain, bestL2);
var theirs = MlNet.Lbfgs(xFull, yTrain, (float)bestL2);
var pMine = encFull.Encode(test).Select(mine.Probability).ToArray();
var pTheirs = theirs(encFull.Encode(test));
double maxDiff = pMine.Zip(pTheirs, (a, b) => Math.Abs(a - b)).Max();
double meanDiff = pMine.Zip(pTheirs, (a, b) => Math.Abs(a - b)).Average();
Console.WriteLine($"  Newton converged in {mine.Iterations} iterations");
Console.WriteLine($"  test-set probabilities: max |diff| {maxDiff:0.000000}, mean |diff| {meanDiff:0.0000000}");

// ---- Improvement ------------------------------------------------------------------------
Console.WriteLine("\nImprovements (5-fold CV on train)");
// 1. Class weights: count each churner as neg/pos customers, at the default 0.5 threshold.
var oofWeighted = OutOfFold(LogisticRecipe(bestL2, weighted: true));
Report("+ class weights", "improvement", oofWeighted, 0.5, "Churners weighted about 2.8x so the loss stops favouring 'stays'.");

// 2. Threshold tuning: leave the model alone, move the cut-off to the F1-best point.
double tLogit = Metrics.BestF1Threshold(oofLogit, yTrain);
Report("+ tuned threshold", "improvement", oofLogit, tLogit, "Same model; threshold chosen to maximise out-of-fold F1.");

double tTree = Metrics.BestF1Threshold(oofTree, yTrain);
Report("Trees + tuned threshold", "improvement", oofTree, tTree, "FastTree with its own F1-best threshold.");

// 3. Engineered features on top of the tuned threshold.
var oofEng = OutOfFold(LogisticRecipe(bestL2, engineered: true));
double tEng = Metrics.BestF1Threshold(oofEng, yTrain);
Report("+ tuned threshold + engineered features", "improvement", oofEng, tEng, "Adds log(1 + tenure), number of add-on services, and tenure of 6 months or less.");

// Choose the final logistic variant by out-of-fold PR AUC (threshold-free), then use its tuned threshold.
double prPlain = Metrics.AveragePrecision(oofLogit, yTrain), prEng = Metrics.AveragePrecision(oofEng, yTrain);
bool useEngineered = prEng > prPlain + 0.002; // must beat plain by more than rounding noise
double finalThreshold = useEngineered ? tEng : tLogit;
Console.WriteLine($"\nPR AUC plain {prPlain:0.0000} vs engineered {prEng:0.0000}: final model uses {(useEngineered ? "engineered" : "plain")} features, threshold {finalThreshold:0.000}");

// ---- Final evaluation (test set, opened once) --------------------------------------------
Console.WriteLine("\nFinal evaluation on the held-out test set");
var finalEnc = Encoder.Fit(train, useEngineered);
var finalModel = Logistic.Fit(finalEnc.Encode(train), yTrain, bestL2);
var pFinal = finalEnc.Encode(test).Select(finalModel.Probability).ToArray();

var treeFinal = MlNet.FastTree(xFull, yTrain);
var pTreeFinal = treeFinal(encFull.Encode(test));

var testRows = new List<Dictionary<string, object>>();
void TestReport(string name, double[] p, double threshold)
{
    var s = Metrics.Evaluate(p, yTest, threshold);
    testRows.Add(new()
    {
        ["name"] = name, ["threshold"] = R(threshold), ["precision"] = R(s.Precision), ["recall"] = R(s.Recall), ["f1"] = R(s.F1),
        ["rocAuc"] = R(s.RocAuc), ["prAuc"] = R(s.PrAuc), ["logLoss"] = R(s.LogLoss), ["accuracy"] = R(s.Accuracy),
        ["truePos"] = s.TruePos, ["falsePos"] = s.FalsePos, ["falseNeg"] = s.FalseNeg, ["trueNeg"] = s.TrueNeg,
    });
    Console.WriteLine($"  {name,-42} t={threshold:0.000}  P {s.Precision:0.000}  R {s.Recall:0.000}  F1 {s.F1:0.000}  ROC {s.RocAuc:0.000}  PR {s.PrAuc:0.000}  acc {s.Accuracy:0.000}  [TP {s.TruePos} FP {s.FalsePos} FN {s.FalseNeg} TN {s.TrueNeg}]");
}
TestReport("Majority class (nobody churns)", test.Select(_ => churnRate).ToArray(), 0.5);
TestReport("Rule: month-to-month contract", test.Select(c => c.Contract == "Month-to-month" ? 1.0 : 0.0).ToArray(), 0.5);
TestReport("Logistic regression, 0.5 threshold", pFinal, 0.5);
TestReport("Gradient-boosted trees, tuned threshold", pTreeFinal, tTree);
TestReport("Final: logistic, tuned threshold", pFinal, finalThreshold);

// ---- Export -----------------------------------------------------------------------------
var curve = Metrics.Curve(pFinal, yTest, 100).Select(c => new { t = R(c.Threshold), p = R(c.Precision), r = R(c.Recall) }).ToList();
var meta = new Dictionary<string, object>
{
    ["dataset"] = "IBM Telco Customer Churn",
    ["trainRows"] = train.Count,
    ["testRows"] = test.Count,
    ["trainChurnRate"] = R(churnRate),
    ["l2"] = bestL2,
    ["engineeredFeatures"] = useEngineered,
    ["seed"] = Seed,
    ["note"] = "Coefficients are per standard deviation of each input. Threshold chosen by 5-fold CV F1 on the training set.",
};
var exported = ExportedModel.From(finalEnc, finalModel, finalThreshold, meta);
string modelJson = JsonSerializer.Serialize(exported, ExportedModel.Json);
foreach (var path in new[] { Path.Combine(root, "src", "Tenure.Api", "model.json"), Path.Combine(root, "docs", "model.json") })
{
    Directory.CreateDirectory(Path.GetDirectoryName(path)!);
    File.WriteAllText(path, modelJson);
}

var opts = new JsonSerializerOptions { WriteIndented = true };
File.WriteAllText(Path.Combine(resultsDir, "cv.json"), JsonSerializer.Serialize(cv, opts));
File.WriteAllText(Path.Combine(resultsDir, "test.json"), JsonSerializer.Serialize(testRows, opts));
File.WriteAllText(Path.Combine(resultsDir, "l2-search.json"), JsonSerializer.Serialize(l2Search, opts));
File.WriteAllText(Path.Combine(resultsDir, "solver-check.json"), JsonSerializer.Serialize(new
{
    newtonIterations = mine.Iterations, l2 = bestL2, maxAbsProbabilityDiff = maxDiff, meanAbsProbabilityDiff = meanDiff,
}, opts));
File.WriteAllText(Path.Combine(root, "docs", "curve.json"), JsonSerializer.Serialize(curve));
Console.WriteLine("\nWrote results/, src/Tenure.Api/model.json, docs/model.json, docs/curve.json");

Console.WriteLine("\nLargest coefficients (per standard deviation)");
foreach (var f in exported.Features.OrderByDescending(f => Math.Abs(f.Coef)).Take(10))
    Console.WriteLine($"  {f.Name,-45} {f.Coef,8:+0.000;-0.000}");
return 0;

static double R(double v) => Math.Round(v, 4);

static string FindRoot()
{
    var dir = new DirectoryInfo(AppContext.BaseDirectory);
    while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "Tenure.sln"))) dir = dir.Parent;
    return dir?.FullName ?? Directory.GetCurrentDirectory();
}
