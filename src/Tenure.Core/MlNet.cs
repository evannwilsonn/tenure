using Microsoft.ML;
using Microsoft.ML.Data;

namespace Tenure.Core;

/// <summary>
/// ML.NET learners trained on exactly the same encoded features as the hand-written model,
/// so any difference in the results comes from the learner, not the preprocessing.
/// </summary>
public static class MlNet
{
    private sealed class Row
    {
        public bool Label { get; set; }
        [VectorType] public float[] Features { get; set; } = Array.Empty<float>();
    }

    private sealed class Out
    {
        public float Probability { get; set; }
    }

    private static IDataView View(MLContext ml, double[][] x, bool[] y) =>
        ml.Data.LoadFromEnumerable(x.Select((r, i) => new Row { Label = y[i], Features = r.Select(v => (float)v).ToArray() }),
            SchemaDefinition.Create(typeof(Row)).Also(s => s[nameof(Row.Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, x[0].Length)));

    private static T Also<T>(this T t, Action<T> act) { act(t); return t; }

    /// <summary>ML.NET's L-BFGS logistic regression, as a check on the hand-written Newton solver.</summary>
    public static Func<double[][], double[]> Lbfgs(double[][] x, bool[] y, float l2, int seed = 1)
    {
        var ml = new MLContext(seed);
        var trainer = ml.BinaryClassification.Trainers.LbfgsLogisticRegression(new Microsoft.ML.Trainers.LbfgsLogisticRegressionBinaryTrainer.Options
        {
            L1Regularization = 0, L2Regularization = l2, OptimizationTolerance = 1e-10f, MaximumNumberOfIterations = 10000,
        });
        var model = trainer.Fit(View(ml, x, y));
        return xs => Predict(ml, model, xs);
    }

    /// <summary>Gradient-boosted trees (ML.NET FastTree), the non-linear comparison.</summary>
    public static Func<double[][], double[]> FastTree(double[][] x, bool[] y, int leaves = 16, int trees = 200, double rate = 0.05, int minLeaf = 20, int seed = 1)
    {
        var ml = new MLContext(seed);
        var trainer = ml.BinaryClassification.Trainers.FastTree(new Microsoft.ML.Trainers.FastTree.FastTreeBinaryTrainer.Options
        {
            NumberOfLeaves = leaves, NumberOfTrees = trees, LearningRate = rate, MinimumExampleCountPerLeaf = minLeaf, Seed = seed,
        });
        var model = trainer.Fit(View(ml, x, y));
        return xs => Predict(ml, model, xs);
    }

    private static double[] Predict(MLContext ml, ITransformer model, double[][] xs)
    {
        var scored = model.Transform(View(ml, xs, new bool[xs.Length]));
        return ml.Data.CreateEnumerable<Out>(scored, reuseRowObject: false).Select(o => (double)o.Probability).ToArray();
    }
}
