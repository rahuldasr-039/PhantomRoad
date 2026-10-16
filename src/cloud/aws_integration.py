"""
Optional AWS Integration Module for S3 storage, processing, and cloud result retrieval.
Runs 100% locally by default. Strictly uses environment variables and IAM credentials.
"""

from typing import Optional, Dict, Any
import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


class AWSVisionAuditManager:
    """
    Manages optional cloud storage of road video clips and reliability audit reports.
    Zero-config local fallback when AWS credentials or buckets are not provided.
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region_name: Optional[str] = None
    ):
        self.bucket_name = bucket_name or os.getenv("PHANTOMROAD_S3_BUCKET", "")
        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self._s3_client = None
        self._init_client()

    def _init_client(self) -> None:
        """
        Initializes boto3 S3 client using standard credential provider chain (Env, IAM role, AWS config).
        """
        try:
            # Only instantiate client if credentials or bucket are hinted
            if self.bucket_name or os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"):
                self._s3_client = boto3.client("s3", region_name=self.region_name)
        except Exception:
            self._s3_client = None

    def is_cloud_enabled(self) -> bool:
        """
        Returns True if AWS S3 client and destination bucket are operational.
        """
        if not self._s3_client or not self.bucket_name:
            return False
        return True

    def get_status_info(self) -> Dict[str, Any]:
        """
        Returns cloud connection health status for UI display.
        """
        if not self.bucket_name:
            return {
                "enabled": False,
                "status": "OFFLINE_LOCAL_MODE",
                "message": "AWS S3 integration is optional and inactive (no bucket configured). All operations are running locally."
            }

        if not self._s3_client:
            return {
                "enabled": False,
                "status": "CREDENTIALS_MISSING",
                "message": f"Bucket '{self.bucket_name}' specified, but AWS credentials were not found in environment."
            }

        return {
            "enabled": True,
            "status": "CONNECTED",
            "bucket": self.bucket_name,
            "region": self.region_name,
            "message": f"Connected to S3 bucket '{self.bucket_name}' ({self.region_name})."
        }

    def upload_file(
        self,
        local_filepath: str,
        s3_key: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Uploads a local file (video or report) to S3.
        Returns s3 URI (e.g. 's3://bucket/key') or None on local mode / error.
        """
        if not self.is_cloud_enabled():
            return None

        if not os.path.exists(local_filepath):
            raise FileNotFoundError(f"Local file does not exist: {local_filepath}")

        key = s3_key or f"phantomroad/{os.path.basename(local_filepath)}"
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

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

    def generate_presigned_download_url(
        self,
        s3_key: str,
        expiration_sec: int = 3600
    ) -> Optional[str]:
        """
        Generates a secure presigned URL for cloud-hosted result retrieval.
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
        except (BotoCoreError, ClientError) as e:
            return None
