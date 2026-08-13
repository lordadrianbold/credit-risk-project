"""
Shared S3 upload helper.

Uploads are OPTIONAL and controlled by an environment variable, so every
script still works fine for local-only runs (e.g. on your Windows desktop)
with zero AWS setup. Uploads only happen when S3_BUCKET is set.

Local usage (no AWS needed, uploads are skipped):
    python src/audit_data.py

With S3 uploads enabled (e.g. inside the Docker container on AWS):
    export S3_BUCKET=your-bucket-name      # Linux/Mac
    set S3_BUCKET=your-bucket-name         # Windows cmd
    python src/audit_data.py

Optional: set S3_PREFIX to namespace uploads within the bucket, e.g.
"credit-risk-project/" (trailing slash). Defaults to no prefix.

AWS credentials are picked up automatically by boto3 from (in order):
env vars, ~/.aws/credentials, or an ECS/EC2 instance role — nothing to
hardcode here.
"""

import os
from pathlib import Path

S3_BUCKET = os.environ.get("S3_BUCKET")  # None means "uploads disabled"
S3_PREFIX = os.environ.get("S3_PREFIX", "")


def upload_file(local_path, s3_key: str | None = None):
    """
    Upload a local file to S3, if S3_BUCKET is configured.

    local_path: path to the file on disk (str or Path)
    s3_key: destination key in the bucket. Defaults to the file's own name,
            placed under S3_PREFIX.

    Silently does nothing if S3_BUCKET isn't set, so scripts stay usable
    with no AWS setup at all. Upload failures are caught and logged as a
    warning rather than crashing the pipeline.
    """
    if not S3_BUCKET:
        return  # uploads disabled — local-only run

    local_path = Path(local_path)
    if not local_path.exists():
        print(f"[s3_utils] Skipping upload — file not found: {local_path}")
        return

    if s3_key is None:
        s3_key = f"{S3_PREFIX}{local_path.name}"

    try:
        import boto3  # imported lazily so boto3 is only required when uploads are used
        s3 = boto3.client("s3")
        s3.upload_file(str(local_path), S3_BUCKET, s3_key)
        print(f"[s3_utils] Uploaded {local_path} -> s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"[s3_utils] WARNING: upload failed for {local_path}: {e}")


def upload_dir(local_dir, s3_prefix: str | None = None):
    """Upload every file in a local directory (non-recursive) to S3."""
    if not S3_BUCKET:
        return

    local_dir = Path(local_dir)
    if not local_dir.exists():
        print(f"[s3_utils] Skipping upload — directory not found: {local_dir}")
        return

    prefix = s3_prefix if s3_prefix is not None else f"{S3_PREFIX}{local_dir.name}/"
    for file_path in local_dir.iterdir():
        if file_path.is_file():
            upload_file(file_path, s3_key=f"{prefix}{file_path.name}")


