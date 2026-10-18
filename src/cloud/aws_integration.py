"""
Stage 4: Optional AWS Cloud Deployment & S3 Audit Management.
Provides enterprise-grade cloud storage, presigned URLs, and result retrieval using boto3.
Local mode remains the default: zero network calls and zero cloud charges when inactive.
"""

from typing import Optional, Dict, Any, List, Union
import os
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, EndpointConnectionError


class CloudStorageMode(str, Enum):
    """Operational mode for PhantomRoad cloud integration."""
    LOCAL = "local"
    AWS_S3 = "aws_s3"


@dataclass
class CloudAuditBundleResult:
    """Structured report returned after syncing an audit package to S3."""
    run_id: str
    timestamp_iso: str
    bucket: str
    storage_mode: str
    uploaded_files: Dict[str, str] = field(default_factory=dict)
    presigned_urls: Dict[str, str] = field(default_factory=dict)
    manifest_uri: Optional[str] = None
    is_success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AWSVisionAuditManager:
    """
    Manages optional AWS S3 synchronization for dashcam video snippets,
    frame snapshots, and quantitative reliability audit reports.

    Key Guarantees:
    - 100% Local Mode by Default: Zero AWS calls, zero network activity, zero cost.
    - Secure Credential Chain: Supports IAM Roles (EC2, ECS, Lambda), STS AssumeRole,
      and Environment Variables. Never hardcodes secrets.
    - Enterprise S3 Security: Enforces Server-Side Encryption (AES256 or KMS) and
      generates time-bounded presigned URLs for secure sharing.
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region_name: Optional[str] = None,
        prefix: Optional[str] = None,
        storage_mode: Optional[str] = None,
        role_arn: Optional[str] = None,
        encryption: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        s3_client: Optional[Any] = None,
        sts_client: Optional[Any] = None
    ):
        # Determine operational storage mode (local by default)
        env_mode = os.getenv("PHANTOMROAD_STORAGE_MODE", CloudStorageMode.LOCAL.value).lower().strip()
        self.storage_mode = (storage_mode or env_mode).lower().strip()
        if self.storage_mode not in (CloudStorageMode.LOCAL.value, CloudStorageMode.AWS_S3.value):
            self.storage_mode = CloudStorageMode.LOCAL.value

        self.bucket_name = (bucket_name if bucket_name is not None else os.getenv("PHANTOMROAD_S3_BUCKET", "")).strip()
        self.region_name = (region_name or os.getenv("PHANTOMROAD_AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1").strip()
        self.prefix = (prefix or os.getenv("PHANTOMROAD_S3_PREFIX") or "phantomroad/").strip()
        if not self.prefix.endswith("/"):
            self.prefix += "/"

        self.role_arn = (role_arn or os.getenv("PHANTOMROAD_AWS_ROLE_ARN", "")).strip()
        self.encryption = (encryption or os.getenv("PHANTOMROAD_S3_ENCRYPTION") or "AES256").strip()
        self.kms_key_id = (kms_key_id or os.getenv("PHANTOMROAD_KMS_KEY_ID", "")).strip()

        # Injected clients for testing / custom sessions
        self._s3_client = s3_client
        self._sts_client = sts_client
        self._auth_type = "NONE"

        if self._s3_client is None:
            self._init_clients()
        else:
            self._auth_type = "INJECTED_CLIENT"

    def _init_clients(self) -> None:
        """
        Initializes boto3 clients if storage mode is 'aws_s3' and bucket is configured.
        Otherwise keeps clients as None for 100% offline local execution.
        """
        # Strictly local mode if storage_mode is 'local' or bucket_name is empty
        if self.storage_mode == CloudStorageMode.LOCAL.value or not self.bucket_name:
            self._s3_client = None
            self._auth_type = "LOCAL_OFFLINE"
            return

        try:
            # Option A: Assume IAM Role if ARN provided
            if self.role_arn:
                session = self._assume_iam_role(self.role_arn)
                if session:
                    self._s3_client = session.client("s3", region_name=self.region_name)
                    self._auth_type = f"IAM_ROLE_ASSUMED ({self.role_arn.split('/')[-1]})"
                    return

            # Option B: Standard boto3 credential chain (Env vars, IAM Instance Profile, ECS Task Role)
            self._s3_client = boto3.client("s3", region_name=self.region_name)

            if os.getenv("AWS_ACCESS_KEY_ID"):
                self._auth_type = "STATIC_ENV_VARS"
            elif os.getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") or os.getenv("AWS_CONTAINER_CREDENTIALS_FULL_URI"):
                self._auth_type = "ECS_TASK_ROLE"
            else:
                self._auth_type = "INSTANCE_PROFILE_OR_CONFIG"

        except Exception as e:
            self._s3_client = None
            self._auth_type = f"INIT_ERROR: {str(e)}"

    def _assume_iam_role(self, role_arn: str) -> Optional[boto3.Session]:
        """
        Assumes an AWS IAM Role via STS to obtain temporary scoped credentials.
        """
        try:
            sts = self._sts_client or boto3.client("sts", region_name=self.region_name)
            session_name = f"PhantomRoadSession-{int(datetime.now(timezone.utc).timestamp())}"
            response = sts.assume_role(
                RoleArn=role_arn,
                RoleSessionName=session_name,
                DurationSeconds=3600
            )
            creds = response["Credentials"]
            return boto3.Session(
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
                region_name=self.region_name
            )
        except Exception:
            return None

    def is_cloud_enabled(self) -> bool:
        """
        Returns True ONLY if AWS S3 mode is active, bucket is configured, and client is operational.
        """
        return bool(
            self.storage_mode == CloudStorageMode.AWS_S3.value and
            self.bucket_name and
            self._s3_client is not None
        )

    def get_status_info(self) -> Dict[str, Any]:
        """
        Returns comprehensive cloud connection and security telemetry for the UI.
        """
        if self.storage_mode == CloudStorageMode.LOCAL.value:
            return {
                "enabled": False,
                "mode": "LOCAL",
                "status": "OFFLINE_LOCAL_MODE",
                "bucket": None,
                "region": self.region_name,
                "auth_type": "NONE",
                "encryption": "N/A",
                "message": "Local mode active (default). Zero AWS calls or charges. All data stored on local disk."
            }

        if not self.bucket_name:
            return {
                "enabled": False,
                "mode": "AWS_S3",
                "status": "BUCKET_NOT_CONFIGURED",
                "bucket": None,
                "region": self.region_name,
                "auth_type": self._auth_type,
                "encryption": self.encryption,
                "message": "AWS S3 mode selected, but no S3 bucket name was configured. Running locally."
            }

        if not self._s3_client:
            return {
                "enabled": False,
                "mode": "AWS_S3",
                "status": "CREDENTIALS_MISSING",
                "bucket": self.bucket_name,
                "region": self.region_name,
                "auth_type": self._auth_type,
                "encryption": self.encryption,
                "message": f"Target bucket '{self.bucket_name}' set, but AWS credentials could not be initialized."
            }

        return {
            "enabled": True,
            "mode": "AWS_S3",
            "status": "CONNECTED",
            "bucket": self.bucket_name,
            "region": self.region_name,
            "prefix": self.prefix,
            "auth_type": self._auth_type,
            "encryption": self.encryption,
            "message": f"Connected to S3 bucket '{self.bucket_name}' ({self.region_name}) via {self._auth_type}."
        }

    def verify_connectivity(self) -> Dict[str, Any]:
        """
        Performs a lightweight read-only validation check against AWS.
        Never alters or creates cloud resources.
        """
        if not self.is_cloud_enabled():
            return {
                "success": False,
                "status": "OFFLINE",
                "error": "Cloud integration is not active. Switch to AWS_S3 mode and provide bucket."
            }

        try:
            # 1. Verify credentials identity via STS
            sts = self._sts_client or boto3.client("sts", region_name=self.region_name)
            identity = sts.get_caller_identity()

            # 2. Verify S3 bucket accessibility via head_bucket
            self._s3_client.head_bucket(Bucket=self.bucket_name)

            return {
                "success": True,
                "status": "HEALTHY",
                "account_id": identity.get("Account"),
                "arn": identity.get("Arn"),
                "bucket": self.bucket_name,
                "region": self.region_name,
                "encryption": self.encryption
            }
        except (BotoCoreError, ClientError, NoCredentialsError, EndpointConnectionError) as e:
            return {
                "success": False,
                "status": "CONNECTIVITY_FAILED",
                "error": str(e)
            }
        except Exception as e:
            return {
                "success": False,
                "status": "UNEXPECTED_ERROR",
                "error": str(e)
            }

    def _build_extra_args(
        self,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Builds S3 upload parameters including Server-Side Encryption and metadata.
        """
        extra_args: Dict[str, Any] = {}

        if content_type:
            extra_args["ContentType"] = content_type

        # Server-Side Encryption
        if self.encryption == "aws:kms":
            extra_args["ServerSideEncryption"] = "aws:kms"
            if self.kms_key_id:
                extra_args["SSEKMSKeyId"] = self.kms_key_id
        elif self.encryption == "AES256":
            extra_args["ServerSideEncryption"] = "AES256"

        if metadata:
            # S3 user metadata keys should be lowercase strings
            sanitized_meta = {str(k).lower(): str(v) for k, v in metadata.items()}
            extra_args["Metadata"] = sanitized_meta

        return extra_args

    def upload_file(
        self,
        local_filepath: str,
        s3_key: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Uploads a local file (video snippet, frame image, or report) to S3.
        Returns the S3 URI ('s3://bucket/key') on success, or None if in local mode / on error.
        """
        if not self.is_cloud_enabled():
            return None

        if not os.path.exists(local_filepath):
            raise FileNotFoundError(f"Local file does not exist: {local_filepath}")

        key = s3_key or f"{self.prefix}{os.path.basename(local_filepath)}"
        extra_args = self._build_extra_args(content_type, metadata)

        try:
            self._s3_client.upload_file(
                Filename=local_filepath,
                Bucket=self.bucket_name,
                Key=key,
                ExtraArgs=extra_args if extra_args else None
            )
            return f"s3://{self.bucket_name}/{key}"
        except (BotoCoreError, ClientError, NoCredentialsError) as e:
            return None

    def upload_in_memory_bytes(
        self,
        data_bytes: bytes,
        s3_key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Uploads in-memory binary data (e.g. encoded JPEG/PNG frame) directly to S3.
        """
        if not self.is_cloud_enabled():
            return None

        key = s3_key if s3_key.startswith(self.prefix) else f"{self.prefix}{s3_key.lstrip('/')}"
        extra_args = self._build_extra_args(content_type, metadata)

        try:
            self._s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data_bytes,
                **extra_args
            )
            return f"s3://{self.bucket_name}/{key}"
        except (BotoCoreError, ClientError, NoCredentialsError):
            return None

    def upload_in_memory_json(
        self,
        data: Union[Dict[str, Any], List[Any]],
        s3_key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Serializes and uploads JSON data directly to S3 without creating a temporary file.
        """
        payload = json.dumps(data, indent=2).encode("utf-8")
        return self.upload_in_memory_bytes(
            data_bytes=payload,
            s3_key=s3_key,
            content_type="application/json",
            metadata=metadata
        )

    def upload_audit_bundle(
        self,
        summary_stats: Dict[str, Any],
        json_report_path: Optional[str] = None,
        csv_report_path: Optional[str] = None,
        video_path: Optional[str] = None,
        run_id: Optional[str] = None,
        presigned_expiry_sec: int = 3600
    ) -> CloudAuditBundleResult:
        """
        Uploads a complete audit package (JSON audit, CSV audit, video snippet, and manifest)
        to S3, and generates secure presigned URLs for each asset.
        """
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        active_run_id = run_id or f"run_{timestamp_str}"
        bundle_prefix = f"{self.prefix}audits/{active_run_id}/"

        if not self.is_cloud_enabled():
            return CloudAuditBundleResult(
                run_id=active_run_id,
                timestamp_iso=datetime.now(timezone.utc).isoformat() + "Z",
                bucket=self.bucket_name or "local_mode",
                storage_mode=self.storage_mode,
                is_success=False,
                error_message="AWS S3 cloud storage is not enabled. Operation completed locally."
            )

        uploaded_files: Dict[str, str] = {}
        presigned_urls: Dict[str, str] = {}
        common_meta = {
            "run-id": active_run_id,
            "agent-version": "1.0.0",
            "competition": "OpenCV-AI-2026",
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
        }

        try:
            # 1. Upload JSON Audit Report
            if json_report_path and os.path.exists(json_report_path):
                json_key = f"{bundle_prefix}audit_report.json"
                uri = self.upload_file(
                    json_report_path,
                    s3_key=json_key,
                    content_type="application/json",
                    metadata={**common_meta, "report-type": "json-audit"}
                )
                if uri:
                    uploaded_files["json_report"] = uri
                    url = self.generate_presigned_download_url(json_key, presigned_expiry_sec)
                    if url:
                        presigned_urls["json_report"] = url

            # 2. Upload CSV Audit Report
            if csv_report_path and os.path.exists(csv_report_path):
                csv_key = f"{bundle_prefix}audit_report.csv"
                uri = self.upload_file(
                    csv_report_path,
                    s3_key=csv_key,
                    content_type="text/csv",
                    metadata={**common_meta, "report-type": "csv-audit"}
                )
                if uri:
                    uploaded_files["csv_report"] = uri
                    url = self.generate_presigned_download_url(csv_key, presigned_expiry_sec)
                    if url:
                        presigned_urls["csv_report"] = url

            # 3. Upload Video Clip (if provided)
            if video_path and os.path.exists(video_path):
                ext = os.path.splitext(video_path)[1].lower() or ".mp4"
                video_key = f"{bundle_prefix}processed_feed{ext}"
                content_type = "video/mp4" if ext == ".mp4" else "application/octet-stream"
                uri = self.upload_file(
                    video_path,
                    s3_key=video_key,
                    content_type=content_type,
                    metadata={**common_meta, "report-type": "video-stream"}
                )
                if uri:
                    uploaded_files["video"] = uri
                    url = self.generate_presigned_download_url(video_key, presigned_expiry_sec)
                    if url:
                        presigned_urls["video"] = url

            # 4. Generate and Upload Structured Audit Manifest
            manifest_data = {
                "manifest_version": "1.0",
                "run_id": active_run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat() + "Z",
                "bucket": self.bucket_name,
                "bundle_prefix": bundle_prefix,
                "summary_stats": summary_stats,
                "artifacts": uploaded_files,
                "encryption": self.encryption,
                "presigned_download_links": presigned_urls
            }
            manifest_key = f"{bundle_prefix}manifest.json"
            manifest_uri = self.upload_in_memory_json(
                manifest_data,
                s3_key=manifest_key,
                metadata={**common_meta, "report-type": "audit-manifest"}
            )
            if manifest_uri:
                uploaded_files["manifest"] = manifest_uri
                manifest_url = self.generate_presigned_download_url(manifest_key, presigned_expiry_sec)
                if manifest_url:
                    presigned_urls["manifest"] = manifest_url

            return CloudAuditBundleResult(
                run_id=active_run_id,
                timestamp_iso=datetime.now(timezone.utc).isoformat() + "Z",
                bucket=self.bucket_name,
                storage_mode=self.storage_mode,
                uploaded_files=uploaded_files,
                presigned_urls=presigned_urls,
                manifest_uri=manifest_uri,
                is_success=True
            )

        except Exception as e:
            return CloudAuditBundleResult(
                run_id=active_run_id,
                timestamp_iso=datetime.now(timezone.utc).isoformat() + "Z",
                bucket=self.bucket_name,
                storage_mode=self.storage_mode,
                uploaded_files=uploaded_files,
                is_success=False,
                error_message=str(e)
            )

    def download_file(
        self,
        s3_key: str,
        local_destination: str
    ) -> Optional[str]:
        """
        Downloads an S3 object to a local file path.
        """
        if not self.is_cloud_enabled():
            return None

        try:
            os.makedirs(os.path.dirname(os.path.abspath(local_destination)), exist_ok=True)
            self._s3_client.download_file(
                Bucket=self.bucket_name,
                Key=s3_key,
                Filename=local_destination
            )
            return local_destination
        except (BotoCoreError, ClientError, NoCredentialsError):
            return None

    def generate_presigned_download_url(
        self,
        s3_key: str,
        expiration_sec: int = 3600
    ) -> Optional[str]:
        """
        Generates a secure, temporary presigned GET URL for sharing S3 artifacts.
        No public bucket policy required.
        """
        if not self.is_cloud_enabled():
            return None

        try:
            url = self._s3_client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=expiration_sec
            )
            return url
        except (BotoCoreError, ClientError):
            return None

    def generate_presigned_upload_url(
        self,
        s3_key: str,
        expiration_sec: int = 3600,
        max_content_length: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generates a presigned POST policy allowing edge dashcams to upload directly to S3
        without holding long-lived AWS IAM credentials.
        """
        if not self.is_cloud_enabled():
            return None

        conditions: List[Any] = []
        if max_content_length:
            conditions.append(["content-length-range", 1, max_content_length])

        # Enforce server-side encryption condition if configured
        fields = {}
        if self.encryption == "AES256":
            fields["x-amz-server-side-encryption"] = "AES256"
            conditions.append({"x-amz-server-side-encryption": "AES256"})

        try:
            post_data = self._s3_client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=s3_key,
                Fields=fields if fields else None,
                Conditions=conditions if conditions else None,
                ExpiresIn=expiration_sec
            )
            return post_data
        except (BotoCoreError, ClientError):
            return None

    def list_cloud_audits(
        self,
        prefix: Optional[str] = None,
        max_keys: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Lists stored audit manifests in the S3 bucket.
        """
        if not self.is_cloud_enabled():
            return []

        search_prefix = prefix or f"{self.prefix}audits/"
        try:
            response = self._s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=search_prefix,
                MaxKeys=max_keys
            )
            items = []
            for obj in response.get("Contents", []):
                items.append({
                    "key": obj.get("Key"),
                    "size_bytes": obj.get("Size"),
                    "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else None,
                    "storage_class": obj.get("StorageClass")
                })
            return items
        except (BotoCoreError, ClientError, NoCredentialsError):
            return []
