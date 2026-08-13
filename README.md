# Credit Risk / Loan Default Prediction Project

A week-by-week project predicting loan default risk on the Lending Club
dataset, moving from data cleaning through classical ML, tree-based models,
interpretation, and a deployed, monitored API.

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
├── notes.md                 # your running log — fill this in as you go
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
   ```

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

3. To run a single script instead of the full week 1-2 pipeline:
   ```
   docker run -v "%cd%\data\raw:/app/data/raw" credit-risk python src/check_correlations.py
   ```

## Optional: S3 uploads

`src/s3_utils.py` uploads key outputs (the cleaned CSV, EDA plots, and the
missingness report) to S3 — but only if the `S3_BUCKET` environment variable
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

3. **Run it as an ECS Fargate task**: create a cluster (Fargate type), a task
   definition pointing at your ECR image, and set these environment
   variables on the task definition:
   ```
   S3_BUCKET=your-bucket-name
   S3_RAW_DATA_KEY=data/raw/lending_club_loans.csv   (optional — this is the default)
   ```
   `entrypoint.sh` runs automatically on container start: it checks whether
   the raw CSV is already present locally (it won't be, on Fargate), and if
   not, downloads it from S3 before the pipeline scripts run. No manual step
   needed beyond setting those two environment variables.

   For the task's IAM role, grant `s3:GetObject` on the raw data path and
   `s3:PutObject` on wherever you want outputs written — scoped to just your
   bucket, not a broad S3 admin policy.

4. **Simpler alternative**: launch a small EC2 instance, install Docker on
   it, `scp` this project folder over, and run the same `docker build` /
   `docker run` commands directly on the instance — you can still mount a
   local file there instead of going through S3, if you `scp` the CSV over too.

## Keep notes as you go

Fill in `notes.md` after each week. It's the single most reusable artifact
in this project — it becomes your README write-up and your answer to
"walk me through a project" in interviews.
