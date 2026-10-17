namespace Tenure.Core;

public sealed record Scores(
    double Accuracy, double Precision, double Recall, double F1,
    double RocAuc, double PrAuc, double LogLoss, double Threshold,
    int TruePos, int FalsePos, int FalseNeg, int TrueNeg);

public static class Metrics
{
    /// <summary>Everything at one threshold, plus the threshold-free ROC AUC and PR AUC.</summary>
    public static Scores Evaluate(IReadOnlyList<double> p, IReadOnlyList<bool> y, double threshold)
    {
        int tp = 0, fp = 0, fn = 0, tn = 0;
        double ll = 0;
        for (int i = 0; i < p.Count; i++)
        {
            bool pred = p[i] >= threshold;
            if (pred && y[i]) tp++; else if (pred) fp++; else if (y[i]) fn++; else tn++;
            double q = Math.Clamp(p[i], 1e-12, 1 - 1e-12);
            ll -= y[i] ? Math.Log(q) : Math.Log(1 - q);
        }
        double precision = tp + fp == 0 ? 0 : (double)tp / (tp + fp);
        double recall = tp + fn == 0 ? 0 : (double)tp / (tp + fn);
        double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
        return new Scores((double)(tp + tn) / p.Count, precision, recall, f1, RocAuc(p, y), AveragePrecision(p, y),
            ll / p.Count, threshold, tp, fp, fn, tn);
    }

    /// <summary>ROC AUC as the chance a random churner scores above a random stayer (ties count half).</summary>
    public static double RocAuc(IReadOnlyList<double> p, IReadOnlyList<bool> y)
    {
        var order = Enumerable.Range(0, p.Count).OrderBy(i => p[i]).ToArray();
        var rank = new double[p.Count];
        for (int i = 0; i < order.Length;)
        {
            int j = i;
            while (j + 1 < order.Length && p[order[j + 1]] == p[order[i]]) j++;
            double avg = (i + j) / 2.0 + 1;
            for (int k = i; k <= j; k++) rank[order[k]] = avg;
            i = j + 1;
        }
        double nPos = y.Count(v => v), nNeg = y.Count - nPos;
        double sumPos = Enumerable.Range(0, p.Count).Where(i => y[i]).Sum(i => rank[i]);
        return nPos == 0 || nNeg == 0 ? double.NaN : (sumPos - nPos * (nPos + 1) / 2) / (nPos * nNeg);
    }

    /// <summary>
    /// Average precision (area under the precision-recall curve). With 26.5% churners, a
    /// random ranking scores about 0.265 here, which makes it a more honest yardstick than ROC AUC.
    /// </summary>
    public static double AveragePrecision(IReadOnlyList<double> p, IReadOnlyList<bool> y)
    {
        // Tied scores are one step of the curve: a constant score gets exactly the base rate,
        // whatever order the ties happen to be sorted in.
        var order = Enumerable.Range(0, p.Count).OrderByDescending(i => p[i]).ToArray();
        double nPos = y.Count(v => v), tp = 0, ap = 0;
        for (int k = 0; k < order.Length;)
        {
            int end = k;
            double groupPos = 0;
            while (end < order.Length && p[order[end]] == p[order[k]]) { if (y[order[end]]) groupPos++; end++; }
            tp += groupPos;
            ap += groupPos * (tp / end); // recall step × precision at this cut-off
            k = end;
        }
        return nPos == 0 ? double.NaN : ap / nPos;
    }

    /// <summary>The threshold that maximizes F1 over the given scores (checked at every distinct score).</summary>
    public static double BestF1Threshold(IReadOnlyList<double> p, IReadOnlyList<bool> y)
    {
        var order = Enumerable.Range(0, p.Count).OrderByDescending(i => p[i]).ToArray();
        double nPos = y.Count(v => v), tp = 0, fp = 0, best = -1, bestT = 0.5;
        for (int k = 0; k < order.Length; k++)
        {
            if (y[order[k]]) tp++; else fp++;
            if (k + 1 < order.Length && p[order[k + 1]] == p[order[k]]) continue;
            double prec = tp / (tp + fp), rec = tp / nPos, f1 = prec + rec == 0 ? 0 : 2 * prec * rec / (prec + rec);
            if (f1 > best) { best = f1; bestT = p[order[k]]; }
        }
        return bestT;
    }

    /// <summary>Precision and recall at each threshold step, for the demo's threshold slider.</summary>
    public static List<(double Threshold, double Precision, double Recall)> Curve(IReadOnlyList<double> p, IReadOnlyList<bool> y, int steps = 100)
    {
        var pts = new List<(double, double, double)>();
        double nPos = y.Count(v => v);
        for (int s = 1; s < steps; s++)
        {
            double t = (double)s / steps;
            int tp = 0, fp = 0;
            for (int i = 0; i < p.Count; i++) if (p[i] >= t) { if (y[i]) tp++; else fp++; }
            pts.Add((t, tp + fp == 0 ? 1 : (double)tp / (tp + fp), tp / nPos));
        }
        return pts;
    }
}
