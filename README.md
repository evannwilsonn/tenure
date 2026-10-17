# Tenure

Predicting which telecom customers are about to cancel, in C# with ML.NET, served from an ASP.NET minimal API.

**[Try the demo](https://evannwilsonn.github.io/tenure/)**: pick a customer, change anything, and see the risk and the reasons behind it.

On 1,409 customers held back from training, the final model catches **77% of the customers who leave** with **54% precision**, an **F1 of 0.635**. A one-line rule ("flag every month-to-month customer") gets 0.584. Predicting that nobody leaves is 73.5% accurate and catches no one.

The project follows seven steps: problem, data, baseline, model, evaluation, improvement and deployment.

---

## 1. Problem

A phone and internet company wants to know which customers are likely to cancel, so a retention team can contact them first. That's binary classification, where:

- **Churn = Yes** is the class that matters, and it's the minority (26.5%).
- A missed leaver (false negative) costs a customer. A wasted call (false positive) costs a few minutes. So recall matters more than precision, but flagging everyone is useless.
- The team needs a reason to open the call with, not just a score. That pushes toward a model whose output can be explained.

**Metric: F1 on the churn class.** It balances precision and recall and can't be gamed by predicting the majority. ROC AUC, PR AUC (average precision) and log loss are reported alongside it. Accuracy is shown only to make the point that it misleads here.

## 2. Data

[IBM Telco Customer Churn](data/README.md): 7,043 customers, 19 inputs, and `Churn`. The inputs are:

- **Account:** tenure in months, contract, payment method, paperless billing, monthly and total charges.
- **Services:** phone, multiple lines, internet type, and six add-ons.
- **Household:** gender, senior citizen, partner, dependents.

What the data needed:

- **11 blank `TotalCharges`.** Every one has tenure 0: they joined this month and haven't been billed. They're read as 0 rather than dropped ([`Data.cs`](src/Tenure.Core/Data.cs)).
- **Categorical inputs** become one dummy column per level, minus a reference level.
- **Every column is standardized with the training set's mean and standard deviation.** That makes coefficients comparable, and coefficient × value reads as "how far this pushes the customer from average".
- **A stratified 80/20 split** (seed 2026) gives 5,634 training and 1,409 test customers, both at 26.54% churn.
- **All model choices use stratified 5-fold cross-validation on the training set.** The test set is opened once, at the end.

The CSV isn't committed, because the source repository doesn't state a license. `scripts/get-data.sh` downloads it.

## 3. Baseline

Two baselines, both scored with the same 5-fold CV:

| Baseline | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Majority class: nobody churns | 0.000 | 0.000 | **0.000** | 0.735 |
| Rule: flag every month-to-month customer | 0.426 | 0.881 | **0.574** | 0.653 |

The month-to-month rule is the real bar. It needs no data science and already catches 88% of leavers, at the cost of flagging more than half of all customers.

## 4. Model

**Logistic regression written from scratch** ([`Model.cs`](src/Tenure.Core/Model.cs)):

- Fitted by Newton's method (iteratively reweighted least squares), with each step solved by a Cholesky decomposition.
- L2 penalty on the weights, not the intercept, chosen by CV log loss from {0.01, 0.1, 1, 10, 30, 100}. λ = 1 won.
- Optional per-row weights, used for the class-weighting experiment.

**Checked against ML.NET.** `LbfgsLogisticRegression`, trained on the same features with the same penalty, agrees on every test-set probability to within **0.000016**. Newton converges in 8 iterations.

**Compared with gradient-boosted trees.** ML.NET `FastTree` used 200 trees, 16 leaves and a learning rate of 0.05.

| 5-fold CV, threshold 0.5 | Precision | Recall | F1 (± fold SD) | ROC AUC | PR AUC |
|---|---|---|---|---|---|
| Logistic regression | 0.659 | 0.557 | 0.604 (± 0.016) | 0.845 | 0.656 |
| Gradient-boosted trees | 0.658 | 0.510 | 0.574 (± 0.026) | 0.840 | 0.651 |

Logistic regression beats the rule on F1 by 0.03. More importantly, it ranks customers far better: ROC AUC is 0.845 against 0.726. At a 0.5 threshold, though, it misses 44% of leavers. Trees don't beat it on this data.

## 5. Evaluation

Every number in sections 3, 4 and 6 is out-of-fold. Each training customer is scored by a model that never saw them, and the fold-to-fold spread of F1 is reported so small differences can be judged against the noise. The held-out test set was used once, after every choice below was fixed:

| Test set, 1,409 customers | Threshold | Precision | Recall | F1 | ROC AUC | Accuracy |
|---|---|---|---|---|---|---|
| Majority class | 0.50 | 0.000 | 0.000 | 0.000 | 0.500 | 0.735 |
| Month-to-month rule | 0.50 | 0.432 | 0.904 | 0.584 | 0.737 | 0.659 |
| Logistic regression | 0.50 | 0.673 | 0.529 | 0.593 | 0.855 | 0.807 |
| Gradient-boosted trees, tuned threshold | 0.29 | 0.535 | 0.751 | 0.625 | 0.845 | 0.761 |
| **Final: logistic, tuned threshold, engineered inputs** | **0.31** | **0.543** | **0.765** | **0.635** | **0.855** | **0.766** |

The confusion matrix of the final model is in [`results/test.json`](results/test.json). It flags 527 customers: 286 do leave and 241 don't. It misses 88 of the 374 leavers. Test numbers sit close to the CV numbers (F1 0.640 in CV, 0.635 on test), so the choices didn't overfit the folds.

What the model learned, per standard deviation of each input:

- **Raises the odds of churn:** short tenure (the strongest effect, mostly through log-tenure), fibre-optic internet, paying by electronic check, and paperless billing.
- **Lowers them:** one- and two-year contracts.

Higher monthly charges get a *negative* coefficient once fibre and the add-ons are in the model. Expensive plans are mostly fibre plus add-ons, and those carry the risk.

## 6. Improvement

Each change was scored with the same 5-fold CV:

| Change | Threshold | Precision | Recall | F1 (± fold SD) | PR AUC |
|---|---|---|---|---|---|
| Logistic regression (start) | 0.50 | 0.659 | 0.557 | 0.604 (± 0.016) | 0.656 |
| + class weights (churners × 2.8) | 0.50 | 0.514 | 0.804 | 0.627 (± 0.018) | 0.656 |
| + tuned threshold instead | 0.31 | 0.547 | 0.762 | 0.637 (± 0.022) | 0.656 |
| + engineered inputs | 0.31 | 0.553 | 0.759 | **0.640** (± 0.019) | **0.670** |

What changed and why:

1. **Class imbalance is a threshold problem first.** The model's probabilities were fine (unchanged ROC and PR AUC), but a 0.5 cut-off is wrong when only 27% of customers churn. Weighting churners 2.8× pushes the cut-off down indirectly. Tuning the threshold to the F1-best point on out-of-fold scores does it directly and scores higher. That was most of the gain: **+0.033 F1**.
2. **Engineered inputs** were log(1 + tenure), a count of add-on services, and a tenure ≤ 6 months flag. Risk falls fastest in the first months, which a straight line in tenure can't capture. They lift PR AUC by 0.014, but F1 by only 0.003, which is inside the fold noise. They were kept because the deciding metric was set in advance: PR AUC, threshold-free, with a minimum gain of 0.002.
3. **Trees with their own tuned threshold** reached 0.628. That's close, but not better, and their reasons can't be read off a coefficient table, so the logistic model is the one deployed.

The full run is one command and writes [`results/`](results):

```sh
./scripts/get-data.sh
dotnet run --project src/Tenure.Train -c Release
```

## 7. Deployment

**ASP.NET minimal API** ([`src/Tenure.Api`](src/Tenure.Api)). It loads the exported model (`model.json`: intercept, threshold, and each input's mean, sd and coefficient) and scores one customer per request:

```sh
dotnet run --project src/Tenure.Api
curl -X POST localhost:5102/predict -H "Content-Type: application/json" -d @src/Tenure.Api/sample.json
```

```json
{
  "probability": 0.7958,
  "churn": true,
  "threshold": 0.3062,
  "raisesRisk": [
    { "input": "Tenure", "value": "2 months", "effect": 1.3535 },
    { "input": "InternetService", "value": "Fiber optic", "effect": 0.6816 },
    { "input": "Contract", "value": "Month-to-month", "effect": 0.5414 }
  ],
  "lowersRisk": [ { "input": "MonthlyCharges", "value": "$85.50", "effect": -0.198 } ]
}
```

How the API behaves:

- **Reasons:** each input's effect on the log-odds relative to an average training customer. Dummy and engineered columns are summed back into the input a person would recognise.
- **Bad input:** missing fields and impossible combinations (add-ons without internet, an unknown contract type) return a 400 with every problem listed.
- **Endpoints:** `GET /health` reports the threshold. [`requests.http`](src/Tenure.Api/requests.http) has a ready request, and there's a [`Dockerfile`](Dockerfile).

**Browser demo** ([`docs/`](docs), on GitHub Pages):

- It runs the same model in JavaScript from the same `model.json`. It gives 0.7958 for the sample customer above, the same as the API.
- A threshold slider shows the precision/recall trade-off on the test set.
- To rebuild it: `python3 scripts/build-demo.py`.

## Tests

```sh
dotnet test   # 19 tests
```

What they check:

- **Loader and splits:** blank `TotalCharges`, and that stratified splits and folds keep the churn rate with no overlap.
- **Newton solver:** it matches the closed-form answer for a single 0/1 input, the gradient is zero at the penalized optimum, and row weights act like duplicated rows.
- **Metrics:** each metric against hand-worked examples, including average precision with tied scores.
- **Export and API:** export round-trip, the predictor's reasons, validation, and the API started in memory.

## Layout

```
src/Tenure.Core    data loading, encoder, Newton logistic regression, metrics, predictor, ML.NET wrappers
src/Tenure.Train   the full experiment: baselines → models → CV → improvement → one test evaluation → export
src/Tenure.Api     minimal API serving model.json
tests/             xUnit
web/, docs/        demo template and the built GitHub Pages site
results/           CV and test metrics written by Tenure.Train
```

.NET 8 · ML.NET 3.0 · xUnit
