# Use a slim Python base image to keep the image small
FROM python:3.11-slim

# System dependencies:
# - build-essential/gcc/g++ for compiling ML libraries (xgboost, shap)
# - curl/unzip needed to install the AWS CLI, which the entrypoint uses to
#   download the raw dataset from S3 when running on ECS/EC2 with no local mount
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install the AWS CLI v2 (used by entrypoint.sh to pull the raw CSV from S3)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip -q awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer
# (only re-runs if requirements.txt changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Make sure output folders exist even before scripts create files in them
RUN mkdir -p data/raw data/processed results/eda_plots

# Make the entrypoint executable
RUN chmod +x entrypoint.sh

# Optional S3 upload/download: set at `docker run` time with
#   docker run -e S3_BUCKET=your-bucket-name <image>
# When unset, scripts run exactly as before, local-file-only, no AWS needed —
# as long as data/raw/lending_club_loans.csv is mounted in (see README).
#
# entrypoint.sh runs first on every container start: it downloads the raw
# CSV from S3 if it's not already present locally, then hands off to CMD.
ENTRYPOINT ["./entrypoint.sh"]

# Default: run the full week 1-2 pipeline in order.
# Override this at `docker run` time to run a single script instead, e.g.:
#   docker run <image> python src/check_correlations.py
CMD ["sh", "-c", "\
    python src/audit_data.py && \
    python src/clean_data.py && \
    python src/explore_features.py && \
    python src/check_correlations.py \
"]


