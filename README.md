# Lending Club Loan Default Prediction

A credit risk model comparing five approaches — logistic regression,
random forest (class-weighted and SMOTE), and XGBoost (class-weighted and
SMOTE) — on real Lending Club data (1,048,575 raw loans, 425,835 resolved
after filtering, 22.1% default rate), with SHAP interpretation, a
business-costed decision threshold, and a full deployment verified live
on AWS Fargate.

## Result

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic regression | 0.549 | 0.086 | 0.149 | 0.705 | 0.393 |
| Random Forest (class-weighted) | 0.340 | 0.669 | 0.451 | 0.709 | 0.401 |
| Random Forest (SMOTE) | 0.396 | 0.383 | 0.390 | 0.696 | 0.380 |
| **XGBoost (class-weighted)** | 0.346 | 0.675 | 0.457 | 0.716 | **0.411** |
| XGBoost (SMOTE) | 0.527 | 0.142 | 0.223 | 0.707 | 0.400 |

**XGBoost (class-weighted) wins on PR-AUC** — the metric that matters
most given the ~22% class imbalance. More interesting: **class weighting
consistently beat SMOTE** on both Random Forest and XGBoost (SMOTE
recall drops to 0.383 and 0.142 vs. 0.669 and 0.675) — a clear,
reproducible finding, not a coin flip.

With SHAP: `int_rate` dominates feature importance by more than 4x the
next feature — expected, since it's the lender's own risk assessment at
origination compressed into one number. A deliberately-chosen decision
threshold (0.410 instead of the default 0.5, based on a stated 5x cost
ratio for a missed default vs. a wrongly-rejected loan) raises recall
from 67.5% to 82.6% while *lowering* total business cost — a real,
actionable finding, not just a metric trade-off.

## Deployment: verified live on AWS, three real bugs found and fixed

The full pipeline — download from S3, clean 425K rows, EDA, all 5
models, SHAP, threshold selection — runs successfully as a containerized
batch job on AWS Fargate, confirmed via complete CloudWatch log
inspection in ~6 minutes end-to-end. Getting there surfaced three
genuine deployment bugs, each diagnosed with direct evidence rather than
guesswork:

1. **A Windows/Linux case-sensitivity bug** — model script files were
   saved with a capitalized first letter (`Model_logistic.py` instead of
   `model_logistic.py`). Completely invisible on Windows (case-insensitive
   filesystem) but broke immediately and clearly inside the Linux
   container Docker actually runs (`ModuleNotFoundError`).
2. **A stale Dockerfile** that silently ran only the original 4-script
   pipeline even after 3 more scripts were added — the container exited
   cleanly with real (partial) output, giving zero indication anything
   was missing. Diagnosed by comparing logs across identical failure
   points on two separate runs. **A wrong hypothesis (out-of-memory) was
   tested and correctly ruled out along the way** — doubling task memory
   from 8GB to 16GB changed nothing about where the failure occurred,
   which was itself the evidence that pointed toward the real cause.
3. **A missing IAM permission** (`logs:CreateLogGroup`) that only
   affected a brand-new log group, not one already used by another
   project's task — fixed with a one-line manual log group creation.

Full details, including exactly how each bug was diagnosed, are in
[`notes.md`](notes.md#deployment-docker--aws-three-real-bugs-found-and-fixed).

**One known limitation, left open rather than hidden**: S3 output
namespacing doesn't fully work as intended — see
[`notes.md`](notes.md#known-limitation-s3-output-namespacing-doesnt-actually-work-as-intended)
for what happened and why it wasn't a real collision.

## Project structure

```
credit-risk-project/
├── data/
│   ├── raw/               # put lending_club_loans.csv here (not included — see below)
│   └── processed/         # cleaned dataset lands here
├── src/                   # all scripts, run from the project root
├── results/
│   └── eda_plots/          # saved default-rate-by-feature charts
├── notebooks/              # optional space for interactive exploration
├── notes.md                 # full week-by-week write-up and findings
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## Getting the data

This project does not include `lending_club_loans.csv` — download it yourself
from Kaggle (search "Lending Club Loan Data") and place it at:
```
data/raw/lending_club_loans.csv
```
If your downloaded file has a different name, either rename it or edit the
`RAW_PATH` variable at the top of `audit_data.py` and `clean_data.py`.

## Local setup (no Docker)

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   ```
   Windows Command Prompt: `venv\Scripts\activate`
   Windows PowerShell: `.\venv\Scripts\Activate.ps1`
   (if blocked, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once)

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   xgboost ships prebuilt wheels for Windows and should install cleanly.
   If `shap` fails to build, install "Build Tools for Visual Studio" (the
   "Desktop development with C++" workload) and retry.

3. Run scripts from the project root, in order:
   ```
   python src/audit_data.py
   python src/clean_data.py
   python src/explore_features.py
   python src/check_correlations.py
   python src/model_logistic.py
   python src/model_trees.py
   python src/model_interpret.py
   ```
   **Note**: on Windows, watch for file casing — every filename here must
   be exact lowercase, since a Docker container later reads these same
   files on a case-sensitive Linux filesystem where Windows won't warn
   you about a mismatch.

## Running with Docker

1. Install Docker Desktop, then build the image from the project root:
   ```
   docker build -t credit-risk .
   ```

2. The raw CSV is intentionally **not** baked into the image (it's large and
   shouldn't live in a Docker image). Mount your local `data/raw` folder
   into the container at run time instead:
   ```
   docker run -v "%cd%\data\raw:/app/data/raw" credit-risk
   ```
   (On Mac/Linux, use `$(pwd)/data/raw:/app/data/raw` instead of `%cd%`.)

   `entrypoint.sh` runs automatically before the pipeline: it checks for the
   CSV locally first (found here, since you mounted it), so no S3 download
   is triggered. S3 download only kicks in when `S3_BUCKET` is set AND no
   local file is found — which is exactly the case on AWS (see below).

3. To run a single script instead of the full week 1-5 pipeline:
   ```
   docker run -v "%cd%\data\raw:/app/data/raw" credit-risk python src/model_interpret.py
   ```

## Optional: S3 uploads

`src/s3_utils.py` uploads key outputs (the cleaned CSV, EDA plots, and
model results) to S3 — but only if the `S3_BUCKET` environment variable
is set. With it unset, every script behaves exactly as before, local-only.

To enable it:
```
# Linux/Mac
export S3_BUCKET=your-bucket-name

# Windows cmd
set S3_BUCKET=your-bucket-name
```
Or when running the container:
```
docker run -e S3_BUCKET=your-bucket-name -v "%cd%\data\raw:/app/data/raw" credit-risk
```
AWS credentials are picked up automatically by boto3 — from environment
variables, `~/.aws/credentials`, or (on ECS/EC2) the instance's IAM role.
Nothing needs to be hardcoded in the code.

**Known gap** (see `notes.md` for details): the `S3_PREFIX` env var only
applies to uploads that don't specify an explicit `s3_key` — every script
here does specify one, so `S3_PREFIX` currently has no effect. Outputs
land at the bucket root.

## Deploying on AWS

1. **Push the image to ECR:**
   ```
   aws ecr create-repository --repository-name credit-risk-project
   aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
   docker tag credit-risk:latest <account-id>.dkr.ecr.<region>.amazonaws.com/credit-risk-project:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/credit-risk-project:latest
   ```

2. **Upload the raw dataset to S3 once** (ECS Fargate tasks have no access to
   your laptop's disk, so there's nothing to mount — this replaces the
   `docker run -v ...` approach you use locally):
   ```
   aws s3 cp data/raw/lending_club_loans.csv s3://your-bucket-name/data/raw/lending_club_loans.csv
   ```

3. **Register the task definition directly via CLI JSON**, rather than the
   console wizard — more reliable and exactly reproducible. Needs:
   `taskRoleArn` (for the container's own S3 access — must be the actual
   task role, not the execution role, which is a distinct and common
   mix-up), `executionRoleArn` (for ECS to pull the image and write logs),
   and environment variables `S3_BUCKET` and `S3_RAW_DATA_KEY`. See
   `notes.md` for the exact bugs this step surfaced.

4. **Create the CloudWatch log group manually first**, if it doesn't
   already exist:
   ```
   aws logs create-log-group --log-group-name /ecs/your-task-name
   ```
   Don't rely on the task definition's `awslogs-create-group: true` option
   alone — the standard `ecsTaskExecutionRolePolicy` doesn't always include
   `logs:CreateLogGroup`, only `CreateLogStream`/`PutLogEvents`.

5. **Run the task**, then verify with the FULL logs, not just the exit
   code:
   ```
   aws ecs run-task --cluster your-cluster --task-definition your-task-def \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[your-subnet],assignPublicIp=ENABLED}"
   ```
   A clean exit code alone does not prove the full pipeline ran — this
   project's own deployment history is a direct example of why (see
   `notes.md`).

6. **Simpler alternative**: launch a small EC2 instance, install Docker on
   it, `scp` this project folder over, and run the same `docker build` /
   `docker run` commands directly on the instance.

## Notes

Full week-by-week reasoning, real results, and the complete deployment
debugging story — including two hypotheses that were tested and one that
was correctly ruled out — is in [`notes.md`](notes.md).

## Project status

All five weeks of modeling are complete, with a clear, well-supported
finding (class weighting beats SMOTE on this dataset) and a properly
business-costed threshold recommendation. **The full pipeline has been
verified running live on AWS Fargate** against the real 425K-row
dataset — not just locally in Docker — with three genuine deployment
bugs found and fixed along the way. One known limitation (S3 output
namespacing) remains open and is documented rather than hidden.



