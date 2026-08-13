#!/bin/sh
# Entrypoint: makes sure the raw dataset is present before running anything.
#
# Local/Docker-with-mount usage: if data/raw/lending_club_loans.csv already
# exists (e.g. you mounted it with `docker run -v ...`), this script does
# nothing extra and just runs the pipeline.
#
# AWS/ECS usage: ECS Fargate tasks have no access to your laptop's disk, so
# there's nothing to mount. Instead, upload the CSV to S3 once:
#   aws s3 cp data/raw/lending_club_loans.csv s3://your-bucket-name/data/raw/lending_club_loans.csv
# then set these environment variables on the task definition:
#   S3_BUCKET=your-bucket-name
#   S3_RAW_DATA_KEY=data/raw/lending_club_loans.csv   (optional, this is the default)
# and this script will download it automatically before the scripts run.

set -e

RAW_FILE="data/raw/lending_club_loans.csv"
S3_RAW_DATA_KEY="${S3_RAW_DATA_KEY:-data/raw/lending_club_loans.csv}"

if [ -f "$RAW_FILE" ]; then
    echo "[entrypoint] Found $RAW_FILE locally (mounted or already present) — skipping S3 download."
elif [ -n "$S3_BUCKET" ]; then
    echo "[entrypoint] $RAW_FILE not found locally. Downloading from s3://$S3_BUCKET/$S3_RAW_DATA_KEY ..."
    mkdir -p data/raw
    aws s3 cp "s3://$S3_BUCKET/$S3_RAW_DATA_KEY" "$RAW_FILE"
    echo "[entrypoint] Download complete."
else
    echo "[entrypoint] WARNING: $RAW_FILE not found and S3_BUCKET is not set."
    echo "[entrypoint] Either mount the file with 'docker run -v ...' or set S3_BUCKET + S3_RAW_DATA_KEY."
    echo "[entrypoint] Continuing anyway — scripts will fail with FileNotFoundError if the data truly isn't there."
fi

# Hand off to whatever command was passed to the container
# (the default pipeline from CMD, or an override like `python src/check_correlations.py`)
exec "$@"


