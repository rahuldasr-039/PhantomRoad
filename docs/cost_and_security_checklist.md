# PhantomRoad — Cost & Security Checklist (Stage 4)
### Financial Guardrails & Enterprise Security Baseline for Cloud Integration

---

## Part 1: Cloud Cost Breakdown & Estimation Matrix

### 1. AWS Free Tier Eligibility
When operating within AWS Free Tier limits (first 12 months of an account):
- **Amazon S3**: 5 GB of standard storage, 20,000 GET requests, 2,000 PUT requests per month — **FREE**.
- **AWS Lambda**: 1,000,000 free requests per month, 3.2 million seconds of compute time — **FREE**.
- **Amazon CloudWatch**: 10 custom metrics, 5GB log data ingestion — **FREE**.

### 2. Standard AWS Pricing (US East us-east-1)

| AWS Resource | Unit | Standard Rate | PhantomRoad Usage Profile |
| :--- | :--- | :--- | :--- |
| **S3 Standard Storage** | Per GB / month | $0.023 | Audit JSON/CSV reports (< 500 KB per run) |
| **S3 Intelligent-Tiering** | Per GB / month | $0.0125 - $0.023 | Incident video snippets ($5 - 20\text{MB}$) |
| **S3 Glacier Instant Retrieval** | Per GB / month | $0.004 | Long-term archival of verified incidents |
| **S3 PUT / POST requests** | Per 1,000 requests | $0.005 | Uploading reports and manifests |
| **S3 GET requests** | Per 1,000 requests | $0.0004 | Presigned URL downloads |
| **Data Transfer In** | Per GB | **$0.00 (Free)** | Ingesting video and reports into S3 |
| **Data Transfer Out** | First 100 GB / month | **$0.00 (Free)** | Downloading reports or presigned links |

---

### 3. Estimated Monthly Cost Model by Fleet Size

By enforcing **Edge Filtering** (only uploading frames or clips when optical degradation or human escalation is detected), PhantomRoad achieves dramatic cost savings over naive continuous streaming.

| Fleet Size | Daily Driving Time | Raw Video Generated | Degraded Incidents Uploaded (5%) | Monthly S3 Storage | Estimated Monthly Cloud Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Vehicle (Dev/Lab)** | 2 hours / day | $10\text{ GB / day}$ | $0.5\text{ GB / day}$ | $15\text{ GB}$ | **$0.35 / month** *(Free Tier: $0.00)* |
| **10 Vehicles** | 4 hours / day | $200\text{ GB / day}$ | $10\text{ GB / day}$ | $300\text{ GB}$ | **$6.90 / month** |
| **100 Vehicles** | 6 hours / day | $3,000\text{ GB / day}$ | $150\text{ GB / day}$ | $4,500\text{ GB}$ | **$68.50 / month** |

*Note: If all 100 vehicles naively uploaded 100% of clear footage without PhantomRoad's edge filtering, the monthly storage bill would exceed $2,070.00 / month.*

---

### 4. Cost Optimization Best Practices

- [ ] **S3 Lifecycle Expiration Rules**: Configure the bucket to automatically transition raw video files to Glacier Instant Retrieval after 14 days, and expire (delete) them after 60 days. Retain lightweight JSON/CSV audit reports for 365+ days.
- [ ] **Edge Triage Before Upload**: Never upload continuous nominal clear video. Only sync frames flagged with `DEGRADED_CORRECTABLE`, `CRITICAL_OBSTRUCTION`, or `HUMAN_REVIEW_REQUIRED`.
- [ ] **H.264 / H.265 Compression**: Ensure uploaded MP4 snippets are encoded with efficient constant rate factors ($CRF \approx 23$) to minimize bytes transferred.
- [ ] **Set AWS Budget Alerts**: Create an AWS Budget alert in the AWS Billing Console configured to notify your email if monthly forecasted spend exceeds **$5.00**.

---

## Part 2: Security & Compliance Checklist

### 1. Identity & Access Management (IAM)
- [ ] **Principle of Least Privilege (PoLP)**: IAM policies attached to PhantomRoad must strictly limit actions to `s3:PutObject`, `s3:GetObject`, and `s3:ListBucket`.
- [ ] **Omit Destructive Permissions**: The IAM policy must **never** include `s3:DeleteObject`, `s3:DeleteBucket`, or `s3:*`.
- [ ] **No Root Account Keys**: Never use root account AWS credentials.
- [ ] **Role-Based Authentication**: When running in AWS (EC2, ECS, or Lambda), use IAM Roles / Instance Profiles instead of static credentials.
- [ ] **Git Secret Prevention**: Verify that `.env` and all credential files are listed in `.gitignore` and never committed to source control.

### 2. S3 Bucket Security Baseline
- [ ] **Block Public Access**: Verify that **all 4 Amazon S3 Block Public Access settings are enabled**:
  - `BlockPublicAcls: True`
  - `IgnorePublicAcls: True`
  - `BlockPublicPolicy: True`
  - `RestrictPublicBuckets: True`
- [ ] **Encryption at Rest**: Ensure default encryption is configured with **Server-Side Encryption (AES256 or AWS KMS)**.
- [ ] **Enforce TLS / HTTPS in Transit**: Attach a bucket policy denying any non-HTTPS requests (`aws:SecureTransport: false`).
- [ ] **Presigned URLs with Time-Bounds**: Never make objects public. Use presigned URLs with short time-to-live ($TTL \le 3600\text{ seconds}$ / 1 hour).

### 3. Privacy, Anonymization & Data Protection
- [ ] **Dashcam Privacy**: If deploying on public roadways, blur pedestrian faces and vehicle license plates prior to cloud synchronization.
- [ ] **Transient In-Memory Processing**: Frames that are determined to be clear and reliable should be discarded from memory immediately after inference.
- [ ] **No Personal Identifiable Information (PII) in Metadata**: S3 object metadata keys must only contain vehicle IDs, sensor IDs, and quality metrics—never driver names or personal identifiers.

### 4. Logging & Auditability
- [ ] **S3 Server Access Logging**: Enable access logging to an isolated log bucket to track all GET/PUT operations.
- [ ] **AWS CloudTrail**: Enable CloudTrail data events for the S3 bucket to maintain an immutable compliance log of all S3 API invocations.
- [ ] **Cryptographic Checksums**: Every uploaded audit bundle includes SHA-256 hashes in `manifest.json` to prove reports were not altered post-capture.

---

## Part 3: Pre-Deployment Security Verification Script

Run this verification check before enabling cloud mode in production:

```powershell
python -c "
from src.cloud.aws_integration import AWSVisionAuditManager
manager = AWSVisionAuditManager()
status = manager.get_status_info()
print('[CHECK 1] Operational Mode:', status['mode'])
print('[CHECK 2] Cloud Enabled:', status['enabled'])
print('[CHECK 3] Encryption Standard:', status['encryption'])
if status['enabled']:
    conn = manager.verify_connectivity()
    print('[CHECK 4] Connectivity:', conn['status'])
    print('[CHECK 5] IAM Identity ARN:', conn.get('arn', 'N/A'))
else:
    print('[CHECK 4] Status:', 'Local Mode Active (Safe - Zero Cloud Footprint)')
"
```
