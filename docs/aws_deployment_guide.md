# PhantomRoad — AWS Deployment Guide (Stage 4)
### Step-by-Step Setup for Local S3 Sync, Cloud Containers, and Serverless Triage

> [!IMPORTANT]
> **Safety Notice**: PhantomRoad runs **100% locally by default**. You do not need an AWS account or active cloud credentials to use PhantomRoad. Only follow this guide if you explicitly choose to configure optional cloud synchronization.

---

## 1. Prerequisites

- Python 3.11+
- An AWS Account with administrative access to IAM and S3 (only for cloud configuration).
- AWS CLI installed (optional, for manual verification).
- PhantomRoad repository cloned and dependencies installed (`pip install -r requirements.txt`).

---

## 2. Deployment Tiers

Choose the deployment tier that matches your operational needs:

```
┌─────────────────────────────────────────────────────────────────┐
│ Tier 1: Local Workstation with S3 Archiving (Recommended First)  │
│ - Agent runs locally on laptop/in-vehicle hardware              │
│ - Uses S3 strictly for archiving video incidents and audit logs │
│ - Lowest complexity, 100% Free Tier compatible                  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│ Tier 2: Cloud-Hosted Dashboard (AWS EC2 / ECS Fargate)          │
│ - Streamlit dashboard deployed in container on AWS              │
│ - Authenticates via IAM Role / Instance Profile (no keys)       │
│ - Multi-user fleet operations console                           │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│ Tier 3: Event-Driven Serverless Pipeline (S3 -> Lambda / Batch) │
│ - Dashcams upload clips directly to S3 via presigned URLs       │
│ - S3 PutObject event triggers Lambda / AWS Batch triage         │
│ - Scalable to thousands of concurrent vehicles                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Tier 1: Local Workstation with S3 Archiving

### Step 1: Create an S3 Bucket
In the AWS S3 Console:
1. Bucket name: e.g., `phantomroad-vision-audit-<your-account-id>` (must be globally unique).
2. AWS Region: Select your preferred region (e.g., `us-east-1`).
3. **Block Public Access**: Keep **all 4 Block Public Access settings ENABLED** (On).
4. **Default Encryption**: Enable Server-side encryption with Amazon S3 managed keys (`SSE-S3` / `AES256`).
5. **Bucket Versioning**: Optional (recommended for audit trail compliance).

### Step 2: Create a Least-Privilege IAM Policy
In AWS IAM > Policies > Create Policy > JSON, paste the following policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PhantomRoadBucketLevelActions",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME_HERE"
    },
    {
      "Sid": "PhantomRoadObjectLevelActions",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME_HERE/phantomroad/*"
    }
  ]
}
```
> [!TIP]
> Notice that `s3:DeleteObject` and wildcard `s3:*` are **omitted**. This guarantees that PhantomRoad cannot delete or tamper with historical audit archives.

### Step 3: Attach Policy to an IAM User or Role
- **For local workstation**: Attach policy to an IAM user and generate Access Keys.
- **For AWS EC2 / ECS**: Attach policy to the EC2 Instance Profile or ECS Task Role (no access keys needed!).

### Step 4: Configure Local Environment Variables
Create or update your `.env` file in the project root:

```env
# Enable AWS S3 Mode (Defaults to 'local' if omitted)
PHANTOMROAD_STORAGE_MODE=aws_s3

# Target S3 Configuration
PHANTOMROAD_S3_BUCKET=phantomroad-vision-audit-your-account-id
PHANTOMROAD_AWS_REGION=us-east-1
PHANTOMROAD_S3_PREFIX=phantomroad/
PHANTOMROAD_S3_ENCRYPTION=AES256

# AWS Credentials (Only required if running outside AWS; do NOT commit to Git!)
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

---

## 4. Tier 2: Cloud Container Deployment (AWS ECS Fargate)

To deploy the PhantomRoad dashboard on AWS ECS Fargate without managing servers:

### Step 1: Build & Push Docker Image to Amazon ECR
```powershell
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# Build Docker image
docker build -t phantomroad:latest .

# Tag and push
docker tag phantomroad:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/phantomroad:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/phantomroad:latest
```

### Step 2: Register ECS Task Definition
PhantomRoad provides a built-in helper to generate the task definition:
```python
from src.cloud.cloud_pipeline import CloudBatchJobDispatcher
task_json = CloudBatchJobDispatcher.generate_ecs_task_definition_template(
    ecr_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/phantomroad:latest",
    cpu="2048",
    memory="4096"
)
```
Register via AWS CLI:
```powershell
aws ecs register-task-definition --cli-input-json file://task_definition.json
```

---

## 5. Tier 3: Serverless Event-Driven Triage (AWS Lambda)

For fleet operators receiving thousands of short dashcam snippets:

```
[In-Vehicle Dashcam]
         │
         │ Upload via Presigned URL (POST)
         ▼
[S3 Ingestion Bucket: s3://raw-videos/]
         │
         │ S3 Event: s3:ObjectCreated:*
         ▼
[AWS Lambda Handler]
         │
         │ 1. Download snippet into /tmp
         │ 2. Compute Laplacian variance & RMS contrast
         │ 3. Run closed-loop recovery if degraded
         ▼
[S3 Audits Bucket: s3://vision-audit/]
  ├── audit_report.json
  └── manifest.json
```

The serverless Lambda handler code is pre-packaged in PhantomRoad:
```python
from src.cloud.cloud_pipeline import CloudBatchJobDispatcher
print(CloudBatchJobDispatcher.generate_lambda_handler_code())
```

---

## 6. S3 Bucket Security Policy (Enforcing TLS 1.2+)

To ensure all data in transit is protected, apply this Bucket Policy to your S3 bucket:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EnforceTLSRequestsOnly",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::YOUR_BUCKET_NAME",
        "arn:aws:s3:::YOUR_BUCKET_NAME/*"
      ],
      "Condition": {
        "Bool": {
          "aws:SecureTransport": "false"
        }
      }
    }
  ]
}
```

---

## 7. Environment Variables Reference

| Variable | Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `PHANTOMROAD_STORAGE_MODE` | `local` | No | Operational mode: `local` (100% offline) or `aws_s3`. |
| `PHANTOMROAD_S3_BUCKET` | *(empty)* | Only for S3 | Name of the destination S3 bucket. |
| `PHANTOMROAD_AWS_REGION` | `us-east-1` | No | AWS region where the bucket resides. |
| `PHANTOMROAD_S3_PREFIX` | `phantomroad/` | No | S3 key prefix for all stored objects. |
| `PHANTOMROAD_S3_ENCRYPTION` | `AES256` | No | Server-side encryption: `AES256` (SSE-S3) or `aws:kms`. |
| `PHANTOMROAD_KMS_KEY_ID` | *(empty)* | Only for KMS | KMS Key ARN or Key ID if using SSE-KMS. |
| `PHANTOMROAD_AWS_ROLE_ARN` | *(empty)* | No | IAM Role ARN to assume via STS for temporary credentials. |
| `AWS_ACCESS_KEY_ID` | *(empty)* | If outside AWS | AWS access key for static credential authentication. |
| `AWS_SECRET_ACCESS_KEY` | *(empty)* | If outside AWS | AWS secret key for static credential authentication. |

---

## 8. Verifying Connectivity Safely

Test your connection from the terminal without creating any cloud resources:

```powershell
python -c "from src.cloud.aws_integration import AWSVisionAuditManager; m = AWSVisionAuditManager(); print(m.get_status_info()); print(m.verify_connectivity())"
```

Or open the **Stage 4: Optional AWS Cloud Deployment & S3 Sync** tab in the Streamlit Operations Dashboard (`python -m streamlit run app.py`) and click **Verify AWS S3 Connectivity**.
