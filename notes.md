# Project Notes: Lending Club Loan Default Prediction

A credit risk model built on real Lending Club data (~1.05M raw loan
records, 425,835 resolved after filtering), comparing five models and
deploying the full pipeline as a containerized batch job on AWS Fargate.

## Week 1: Data understanding + cleaning

- **Target definition**: `default` = 1 for `Charged Off` or `Default`
  loan_status, 0 for `Fully Paid`. All other statuses (`Current`, `Late`,
  `In Grace Period`) excluded — their final outcome isn't known yet, and
  including them would either leak future information or mislabel
  still-open loans.
- **Real scale**: 1,048,575 raw rows, 145 columns → 425,835 resolved loans
  after filtering, 132 columns after dropping known leakage fields.
  **22.1% default rate** — close to, but not identical to, the ~21.5%
  seen in an earlier synthetic test run, which was a nice sanity check
  that the synthetic data generator was realistically shaped.
- **Leakage columns dropped**: post-outcome fields like `total_pymnt`,
  `recoveries`, `last_pymnt_d`, `debt_settlement_flag`, `settlement_status`
  — anything only knowable after a loan's outcome is determined.
- **Known limitation, not yet fully closed**: the real dataset has 145
  columns, far more than the hand-picked `LEAKAGE_COLUMNS` list covers
  (e.g. `out_prncp`, `settlement_date`, `settlement_percentage`,
  `settlement_term` aren't explicitly listed). This isn't a correctness
  bug — `eval_utils.py` only ever selects a small, explicit whitelist of
  columns into the model (`NUMERIC_FEATURES` / `CATEGORICAL_FEATURES`),
  so unlisted leakage columns simply sit unused in `clean_loans.csv`
  rather than leaking into training. But the leakage list itself should
  be treated as "good enough for the features actually used," not as an
  exhaustive audit of the full 145-column schema.
- **Verified with a raw-schema synthetic dataset** (not just the
  already-cleaned one used for Week 3-5 testing) — generated fake data
  matching Lending Club's actual messy shape (text `loan_status` values,
  14 leakage columns, mixed types) specifically to test `audit_data.py`
  and `clean_data.py` themselves, which had been written but never
  actually executed until that point.

## Week 2: Exploratory analysis + feature understanding

- `explore_features.py` and `check_correlations.py` ran cleanly against
  the full real dataset — default-rate-by-feature plots for
  annual_inc, dti, loan_amnt, int_rate, revol_util, open_acc, grade,
  purpose, home_ownership, term, plus a correlation matrix and a
  missingness-vs-target report.
- Notable missingness finding: several hardship-related fields (99.3%+
  missing) show a MUCH higher default rate when present (e.g.
  `hardship_loan_status` present → 77.0% default rate vs. 21.8% when
  missing) — unsurprising (hardship program enrollment correlates with
  financial distress) but a genuine, real signal worth being aware of if
  these fields were ever considered as features later.

## Week 3: Feature engineering + logistic regression baseline

- **Engineered feature**: `loan_to_income` (loan_amnt / annual_inc) — a
  simple, interpretable affordability ratio.
- **Real bug found and fixed**: the real Lending Club data stores
  employment length as text (`"10+ years"`, `"< 1 year"`, `"3 years"`,
  `"n/a"`) in a column called `emp_length` — not as a ready-made number.
  An earlier synthetic test dataset had (unintentionally) cut this corner
  by generating a pre-converted numeric `emp_length_years` column
  directly, which let this gap go undetected through the first round of
  "verified" testing. Fixed by adding a `parse_emp_length()` function to
  `eval_utils.py` that correctly handles all the real text formats. This
  is a good example of how synthetic test data can hide a real-world data
  quality issue that only surfaces once real data is used — worth
  remembering as a general lesson, not just a one-off fix.
- **Baseline result** (real data): Precision 0.549, Recall 0.086, F1
  0.149, ROC-AUC 0.705, PR-AUC 0.393. Low recall is expected and even
  appropriate for an unregularized default-threshold logistic baseline on
  imbalanced data — it's the floor the other models are compared against,
  not the deployed choice.
- Top coefficients by magnitude: loan grade (C/D/E/B/F, all positive —
  worse grades correctly predict higher risk), `annual_inc` (negative,
  as expected), `int_rate` (positive).

## Week 4: Tree-based models + class imbalance handling

Compared class weighting vs. SMOTE oversampling head-to-head for both
Random Forest and XGBoost — four real, comparable results rather than
picking one approach blindly.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic regression | 0.549 | 0.086 | 0.149 | 0.705 | 0.393 |
| Random Forest (class-weighted) | 0.340 | 0.669 | 0.451 | 0.709 | 0.401 |
| Random Forest (SMOTE) | 0.396 | 0.383 | 0.390 | 0.696 | 0.380 |
| **XGBoost (class-weighted)** | 0.346 | 0.675 | 0.457 | 0.716 | **0.411** |
| XGBoost (SMOTE) | 0.527 | 0.142 | 0.223 | 0.707 | 0.400 |

- **Clear, consistent finding**: class weighting beat SMOTE on BOTH
  Random Forest and XGBoost — SMOTE variants show noticeably lower recall
  (0.383 and 0.142 vs. 0.669 and 0.675). This is a genuine result, not a
  coin flip: SMOTE's synthetic minority examples apparently didn't help
  either model generalize as well as simply up-weighting the loss on real
  minority-class examples did, on this dataset.
- **XGBoost (class-weighted) wins on PR-AUC** (0.411, the best of all
  five models) — the metric that matters most given the ~22% imbalance,
  since PR-AUC is far more informative than ROC-AUC when the positive
  class is a minority.

## Week 5: Model interpretation + business framing

- **SHAP global importance**: `int_rate` dominates by a wide margin
  (mean |SHAP| = 0.517, more than 4x the next feature) — makes sense,
  since it's the lender's own risk assessment at origination compressed
  into one number. Followed by `loan_to_income`, `term_60_months`,
  `home_ownership` (MORTGAGE/RENT), `open_acc`, `dti`, `revol_util`,
  `emp_length_years`, `annual_inc`.
- **Individual prediction walkthroughs**: highest-risk example
  (predicted probability 0.921) driven by high `int_rate`,
  `purpose_small_business`, grade G, 60-month term. Lowest-risk example
  (probability 0.024) driven by low `int_rate`, low `loan_to_income`, low
  `loan_amnt` — sensible, explainable direction on both ends.
- **Deliberate threshold selection**: computed real confusion-matrix
  costs across a threshold grid (not an approximated formula — an earlier
  version of this logic degenerated to a nonsensical extreme threshold on
  synthetic test data, caught and fixed before this ever ran on real
  data). Assuming a missed default costs 5x a wrongly-rejected good loan:
  moving from the default 0.5 threshold to 0.410 raises recall from 67.5%
  to 82.6%, while total business cost actually *decreases* (54,769 →
  51,949) — a genuine, actionable finding, not just a metric trade-off.

## Deployment: Docker + AWS (three real bugs found and fixed)

Reused existing AWS infrastructure from the time-series forecasting
project (`ts-forecast-cluster`, `ts-forecast-task-role`,
`ecsTaskExecutionRole`, and the same S3 bucket under a
`credit-risk-project/` prefix for the raw data) rather than duplicating
setup — a new ECR repository was the only genuinely new piece of
infrastructure needed.

- **Bug 1 — Windows/Linux case-sensitivity (the sneakiest one)**: model
  script files were saved as `Model_logistic.py`, `Model_trees.py`,
  `Model_interpret.py`, `Eval_utils.py` (capitalized first letter).
  Windows filesystems are case-*insensitive*, so this was completely
  invisible locally — `from eval_utils import ...` "worked" on Windows
  even though the file was actually named `Eval_utils.py`. Docker
  containers run Linux even on a Windows host (via Docker Desktop's Linux
  VM), and Linux filesystems ARE case-sensitive, so this broke immediately
  and clearly (`ModuleNotFoundError: No module named 'eval_utils'`) the
  first time it actually ran in a container. Fixed by renaming all four
  files to exact lowercase — note that a case-only rename doesn't always
  "stick" via Windows Explorer/VS Code's rename dialog, and sometimes
  needs to go through an intermediate name first (e.g. rename to
  `eval_utils_temp.py`, then to `eval_utils.py`) to actually take effect.
- **Bug 2 — stale Dockerfile CMD**: the Dockerfile's `CMD` still only ran
  the original Week 1-2 four-script pipeline, even after Weeks 3-5 were
  built. This produced a genuinely misleading result: the container ran
  successfully, exited with code 0, and uploaded real (partial) output to
  S3 — with ZERO indication anything was missing, since the missing
  scripts were simply never invoked at all, not failing. Diagnosed by
  noticing the CloudWatch logs consistently stopped at the exact same
  point across multiple runs. **A wrong hypothesis was tested and
  correctly ruled out along the way**: initially suspected an
  out-of-memory kill, since the silent stop looked like a `SIGKILL`
  signature (no traceback, no partial output). Doubled task memory from
  8GB to 16GB as a test — the failure point didn't change AT ALL, which
  is actually strong evidence *against* the OOM hypothesis, not for it.
  That observation is what led to checking the actual local Dockerfile
  content directly, which confirmed the real cause.
- **Bug 3 — missing `logs:CreateLogGroup` permission**: `ecsTaskExecutionRole`
  (reused from the forecasting project) could write to an *existing*
  CloudWatch log group but couldn't auto-create a brand-new one
  (`/ecs/credit-risk-task` didn't exist yet, unlike `/ecs/ts-forecast-task`,
  which already existed from earlier debugging on that project). Fixed
  with a one-line `aws logs create-log-group` run ahead of time, using
  the broader IAM user's own permissions rather than the task role's.

## Verified live on AWS — full pipeline, real 425K-row dataset

Confirmed via complete CloudWatch log inspection (not just exit code —
an earlier run's silent partial-success taught that lesson directly):
download (525MB from S3, ~3 seconds) → clean → EDA → all 5 models →
SHAP → threshold selection, entirely on AWS Fargate, **~6 minutes
end-to-end** (22:03:42–22:09:29 UTC). Every real-data result matches the
local Docker run almost exactly (same metrics to 2-3 decimal places),
confirming the containerized environment behaves identically to local —
the earlier failures were entirely about deployment configuration
(file casing, stale image, IAM permissions), never about the actual
modeling code once it was genuinely running.

## Known limitation: S3 output namespacing doesn't actually work as intended

`s3_utils.py`'s `S3_PREFIX` environment variable only applies when
`upload_file()` is called *without* an explicit `s3_key` — but every
script in this project (mirroring the same pattern in the forecasting
project) always passes an explicit `s3_key` (e.g.
`s3_key="results/model_comparison.csv"`), which bypasses `S3_PREFIX`
entirely. This means this project's outputs landed at the bucket root
(`data/processed/clean_loans.csv`, `results/eda_plots/...`,
`results/model_comparison.csv`) rather than under a
`credit-risk-project/` prefix as intended — sitting alongside the
forecasting project's own files in the same shared bucket. No actual
collision happened (different filenames, by luck), but this is a real,
still-open gap: the S3 upload paths would need to be made
prefix-aware (e.g. by removing the hardcoded `s3_key` and letting
`S3_PREFIX` apply, or by explicitly prepending it) to genuinely separate
the two projects' outputs. Flagged here rather than silently left
unaddressed.

## Project status

All five weeks of modeling are complete with a real, well-supported
finding (class weighting beats SMOTE on this dataset; XGBoost
class-weighted wins on PR-AUC) and a properly interpreted, business-costed
threshold recommendation. The full pipeline has been verified running
live on AWS Fargate against the real 425K-row dataset, with three
genuine deployment bugs found, diagnosed with direct evidence, and fixed
— including one case where a wrong hypothesis (OOM) was tested and
correctly ruled out rather than assumed. One known limitation (S3 output
namespacing) remains open and is documented above rather than hidden.



