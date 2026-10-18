"""
Unit and integration tests for Stage 4: Optional AWS Cloud Deployment & S3 Audit Storage.
Verifies local offline default, IAM role / env configuration, S3 file and bundle uploads,
presigned URLs, KMS/AES encryption, connectivity checks, error handling, and cloud dispatching.
All AWS interactions are 100% mocked — zero network calls, zero live resources, zero cost.
"""

import os
import json
import tempfile
from unittest.mock import MagicMock, patch
import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from src.cloud.aws_integration import (
    AWSVisionAuditManager,
    CloudStorageMode,
    CloudAuditBundleResult
)
from src.cloud.cloud_pipeline import (
    CloudProcessingEvent,
    CloudBatchJobDispatcher
)


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def mock_s3():
    """Mock boto3 S3 client with standard method signatures."""
    client = MagicMock()
    client.upload_file.return_value = None
    client.put_object.return_value = {"ETag": '"test-etag"'}
    client.download_file.return_value = None
    client.head_bucket.return_value = {}
    client.generate_presigned_url.return_value = "https://mock-bucket.s3.amazonaws.com/test?token=signed"
    client.generate_presigned_post.return_value = {
        "url": "https://mock-bucket.s3.amazonaws.com",
        "fields": {"key": "test.mp4", "x-amz-signature": "mock"}
    }
    client.list_objects_v2.return_value = {
        "Contents": [
            {
                "Key": "phantomroad/audits/run_01/manifest.json",
                "Size": 1024,
                "LastModified": None,
                "StorageClass": "STANDARD"
            }
        ]
    }
    return client


@pytest.fixture
def mock_sts():
    """Mock boto3 STS client."""
    client = MagicMock()
    client.get_caller_identity.return_value = {
        "Account": "123456789012",
        "Arn": "arn:aws:iam::123456789012:role/PhantomRoadRole",
        "UserId": "AROAEXAMPLE:session"
    }
    client.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ASIA_MOCK_KEY",
            "SecretAccessKey": "MOCK_SECRET",
            "SessionToken": "MOCK_TOKEN",
            "Expiration": None
        }
    }
    return client


@pytest.fixture
def sample_video_file():
    """Creates a temporary sample video file for upload testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41")
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def sample_report_files():
    """Creates temporary JSON and CSV report files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "report.json")
        csv_path = os.path.join(tmpdir, "report.csv")
        with open(json_path, "w") as f:
            json.dump({"total_frames": 20, "reliable_count": 18}, f)
        with open(csv_path, "w") as f:
            f.write("frame,state,score\n0,RELIABLE,0.85\n1,DEGRADED,0.40\n")
        yield json_path, csv_path


# =====================================================================
# 1. LOCAL MODE DEFAULT & GRACEFUL FALLBACK TESTS
# =====================================================================

def test_default_initialization_is_strictly_local():
    """
    Verifies that with zero config, the manager operates 100% locally.
    Never calls boto3 and never errors.
    """
    with patch.dict(os.environ, {}, clear=True):
        manager = AWSVisionAuditManager()
        assert manager.storage_mode == CloudStorageMode.LOCAL.value
        assert not manager.is_cloud_enabled()

        status = manager.get_status_info()
        assert status["mode"] == "LOCAL"
        assert status["status"] == "OFFLINE_LOCAL_MODE"
        assert "Zero AWS calls" in status["message"]

        # All operations must return safe offline fallbacks
        assert manager.upload_file("any.mp4") is None
        assert manager.upload_in_memory_bytes(b"data", "key.bin") is None
        assert manager.upload_in_memory_json({}, "key.json") is None
        assert manager.download_file("key", "dest") is None
        assert manager.generate_presigned_download_url("key") is None
        assert manager.generate_presigned_upload_url("key") is None
        assert manager.list_cloud_audits() == []

        connectivity = manager.verify_connectivity()
        assert connectivity["status"] == "OFFLINE"
        assert connectivity["success"] is False


def test_explicit_storage_mode_local_overrides_credentials():
    """
    Even if AWS environment variables exist, setting PHANTOMROAD_STORAGE_MODE=local
    must strictly force local mode without initializing AWS clients.
    """
    env_vars = {
        "AWS_ACCESS_KEY_ID": "AKIA_MOCK_TEST",
        "AWS_SECRET_ACCESS_KEY": "MOCK_SECRET_KEY",
        "PHANTOMROAD_S3_BUCKET": "my-production-bucket",
        "PHANTOMROAD_STORAGE_MODE": "local"
    }
    with patch.dict(os.environ, env_vars, clear=True):
        manager = AWSVisionAuditManager()
        assert manager.storage_mode == CloudStorageMode.LOCAL.value
        assert not manager.is_cloud_enabled()
        assert manager._s3_client is None


def test_missing_bucket_behavior():
    """
    If AWS_S3 mode is requested but no bucket name is provided,
    manager gracefully flags BUCKET_NOT_CONFIGURED.
    """
    manager = AWSVisionAuditManager(storage_mode="aws_s3", bucket_name="")
    assert not manager.is_cloud_enabled()
    status = manager.get_status_info()
    assert status["status"] == "BUCKET_NOT_CONFIGURED"
    assert status["enabled"] is False


def test_credentials_missing_behavior():
    """
    If AWS_S3 mode is configured with a bucket, but boto3 fails to find credentials,
    manager reports CREDENTIALS_MISSING cleanly without crashing.
    """
    with patch("boto3.client", side_effect=NoCredentialsError()):
        manager = AWSVisionAuditManager(
            storage_mode="aws_s3",
            bucket_name="vision-test-bucket"
        )
        assert not manager.is_cloud_enabled()
        status = manager.get_status_info()
        assert status["status"] == "CREDENTIALS_MISSING"


# =====================================================================
# 2. IAM ROLE & SECURE CREDENTIAL TESTS
# =====================================================================

def test_iam_role_assumption_flow(mock_sts):
    """
    Tests assuming an IAM Role via STS to obtain temporary scoped credentials.
    """
    mock_session = MagicMock()
    mock_s3_instance = MagicMock()
    mock_session.client.return_value = mock_s3_instance

    role_arn = "arn:aws:iam::123456789012:role/PhantomRoadS3SyncRole"

    with patch("boto3.Session", return_value=mock_session):
        manager = AWSVisionAuditManager(
            storage_mode="aws_s3",
            bucket_name="vision-test-bucket",
            role_arn=role_arn,
            sts_client=mock_sts
        )

        assert manager.is_cloud_enabled()
        assert "IAM_ROLE_ASSUMED" in manager._auth_type
        mock_sts.assume_role.assert_called_once()
        call_kwargs = mock_sts.assume_role.call_args[1]
        assert call_kwargs["RoleArn"] == role_arn


# =====================================================================
# 3. S3 UPLOAD & SERVER-SIDE ENCRYPTION TESTS
# =====================================================================

def test_mocked_s3_file_upload(mock_s3, sample_video_file):
    """
    Verifies uploading a local video file with metadata and AES256 encryption.
    """
    manager = AWSVisionAuditManager(
        storage_mode="aws_s3",
        bucket_name="vision-test-bucket",
        encryption="AES256",
        s3_client=mock_s3
    )

    custom_meta = {"camera_id": "front_left", "fps": "24"}
    s3_uri = manager.upload_file(
        local_filepath=sample_video_file,
        s3_key="phantomroad/clips/test_feed.mp4",
        content_type="video/mp4",
        metadata=custom_meta
    )

    assert s3_uri == "s3://vision-test-bucket/phantomroad/clips/test_feed.mp4"
    mock_s3.upload_file.assert_called_once()
    kwargs = mock_s3.upload_file.call_args[1]
    assert kwargs["Bucket"] == "vision-test-bucket"
    assert kwargs["Key"] == "phantomroad/clips/test_feed.mp4"
    assert kwargs["ExtraArgs"]["ContentType"] == "video/mp4"
    assert kwargs["ExtraArgs"]["ServerSideEncryption"] == "AES256"
    assert kwargs["ExtraArgs"]["Metadata"]["camera_id"] == "front_left"


def test_mocked_s3_file_upload_kms_encryption(mock_s3, sample_video_file):
    """
    Verifies KMS encryption headers are attached when aws:kms is configured.
    """
    kms_arn = "arn:aws:kms:us-east-1:123456789012:key/mock-kms-key-id"
    manager = AWSVisionAuditManager(
        storage_mode="aws_s3",
        bucket_name="vision-test-bucket",
        encryption="aws:kms",
        kms_key_id=kms_arn,
        s3_client=mock_s3
    )

    s3_uri = manager.upload_file(sample_video_file)
    assert s3_uri.startswith("s3://vision-test-bucket/phantomroad/")
    kwargs = mock_s3.upload_file.call_args[1]
    assert kwargs["ExtraArgs"]["ServerSideEncryption"] == "aws:kms"
    assert kwargs["ExtraArgs"]["SSEKMSKeyId"] == kms_arn


def test_upload_missing_file_raises_not_found(mock_s3):
    """
    Attempting to upload a nonexistent file must raise FileNotFoundError.
    """
    manager = AWSVisionAuditManager(
        storage_mode="aws_s3",
        bucket_name="vision-test-bucket",
        s3_client=mock_s3
    )
    with pytest.raises(FileNotFoundError):
        manager.upload_file("completely_missing_nonexistent_file.mp4")


def test_upload_client_error_recovery(mock_s3, sample_video_file):
    """
    Verifies that ClientError during S3 upload is caught and returns None safely.
    """
    error_response = {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}
    mock_s3.upload_file.side_effect = ClientError(error_response, "UploadPart")

    manager = AWSVisionAuditManager(
        storage_mode="aws_s3",
        bucket_name="vision-test-bucket",
        s3_client=mock_s3
    )
    result = manager.upload_file(sample_video_file)
    assert result is None


def test_mocked_in_memory_bytes_and_json_upload(mock_s3):
    """
    Tests uploading binary bytes and JSON objects directly from memory.
    """
    manager = AWSVisionAuditManager(
        storage_mode="aws_s3",
        bucket_name="vision-test-bucket",
        s3_client=mock_s3
    )

    # 1. Binary Bytes Upload
    uri_bytes = manager.upload_in_memory_bytes(
        data_bytes=b"fake-jpeg-frame-data",
        s3_key="frames/frame_001.jpg",
        content_type="image/jpeg"
    )
    assert uri_bytes == "s3://vision-test-bucket/phantomroad/frames/frame_001.jpg"

    # 2. JSON Upload
    payload = {"status": "RELIABLE", "laplacian_variance": 142.5}
    uri_json = manager.upload_in_memory_json(
        data=payload,
        s3_key="telemetry/frame_001.json"
    )
    assert uri_json == "s3://vision-test-bucket/phantomroad/telemetry/frame_001.json"
    assert mock_s3.put_object.call_count == 2


# =====================================================================
# 4. AUDIT BUNDLE UPLOAD & MANIFEST TESTS
# =====================================================================

def test_mocked_audit_bundle_upload(mock_s3, sample_report_files, sample_video_file):
    """
    Verifies packaging and uploading complete batch run: JSON, CSV, video, and manifest.
    """
    json_path, csv_path = sample_report_files
    manager = AWSVisionAuditManager(
        storage_mode="aws_s3",
        bucket_name="vision-test-bucket",
        s3_client=mock_s3
    )

    summary = {
        "total_frames_analyzed": 20,
        "reliable_frames": 18,
        "recovery_rate": 0.90
    }

    result = manager.upload_audit_bundle(
        summary_stats=summary,
        json_report_path=json_path,
        csv_report_path=csv_path,
        video_path=sample_video_file,
        run_id="run_test_123"
    )

    assert isinstance(result, CloudAuditBundleResult)
    assert result.is_success is True
    assert result.run_id == "run_test_123"
    assert "json_report" in result.uploaded_files
    assert "csv_report" in result.uploaded_files
    assert "video" in result.uploaded_files
    assert "manifest" in result.uploaded_files

    # Presigned URLs should be generated for each asset
    assert "json_report" in result.presigned_urls
    assert "video" in result.presigned_urls
    assert "manifest" in result.presigned_urls
    assert result.manifest_uri.endswith("manifest.json")


def test_bundle_upload_offline_fallback(sample_report_files):
    """
    In local mode, bundle upload returns clean offline status without errors.
    """
    json_path, csv_path = sample_report_files
    manager = AWSVisionAuditManager(storage_mode="local")
    result = manager.upload_audit_bundle(
        summary_stats={},
        json_report_path=json_path,
        csv_report_path=csv_path
    )
    assert result.is_success is False
    assert "not enabled" in result.error_message


# =====================================================================
# 5. PRESIGNED URLS & DOWNLOAD TESTS
# =====================================================================

def test_mocked_presigned_download_url(mock_s3):
    """
    Tests generating presigned GET URLs with customizable expiration.
    """
    manager = AWSVisionAuditManager(
        storage_mode="aws_s3",
        bucket_name="vision-test-bucket",
        s3_client=mock_s3
    )
    url = manager.generate_presigned_download_url("phantomroad/video.mp4", expiration_sec=7200)
    assert "https://" in url
    mock_s3.generate_presigned_url.assert_called_once_with(
        ClientMethod="get_object",
        Params={"Bucket": "vision-test-bucket", "Key": "phantomroad/video.mp4"},
        ExpiresIn=7200
    )


def test_mocked_presigned_upload_url(mock_s3):
    """
    Tests generating presigned POST data for edge device uploads.
    """
    manager = AWSVisionAuditManager(
        storage_mode="aws_s3",
        bucket_name="vision-test-bucket",
        s3_client=mock_s3
    )
    post_data = manager.generate_presigned_upload_url(
        s3_key="uploads/dashcam_01.mp4",
        max_content_length=100 * 1024 * 1024
    )
    assert "url" in post_data
    assert "fields" in post_data
    mock_s3.generate_presigned_post.assert_called_once()


def test_mocked_download_file(mock_s3):
    """
    Verifies downloading an S3 asset to a local file.
    """
    manager = AWSVisionAuditManager(
        storage_mode="aws_s3",
        bucket_name="vision-test-bucket",
        s3_client=mock_s3
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = os.path.join(tmpdir, "downloads", "result.json")
        out = manager.download_file("phantomroad/reports/audit.json", dest_path)
        assert out == dest_path
        mock_s3.download_file.assert_called_once_with(
            Bucket="vision-test-bucket",
            Key="phantomroad/reports/audit.json",
            Filename=dest_path
        )


def test_mocked_list_cloud_audits(mock_s3):
    """
    Verifies listing stored audits in the S3 bucket.
    """
    manager = AWSVisionAuditManager(
        storage_mode="aws_s3",
        bucket_name="vision-test-bucket",
        s3_client=mock_s3
    )
    items = manager.list_cloud_audits()
    assert len(items) == 1
    assert items[0]["key"] == "phantomroad/audits/run_01/manifest.json"
    assert items[0]["size_bytes"] == 1024


# =====================================================================
# 6. CONNECTIVITY HEALTH CHECKS
# =====================================================================

def test_mocked_connectivity_verification_success(mock_s3, mock_sts):
    """
    Tests non-mutating connectivity health check via STS caller identity and S3 head_bucket.
    """
    manager = AWSVisionAuditManager(
        storage_mode="aws_s3",
        bucket_name="vision-test-bucket",
        s3_client=mock_s3,
        sts_client=mock_sts
    )
    res = manager.verify_connectivity()
    assert res["success"] is True
    assert res["status"] == "HEALTHY"
    assert res["account_id"] == "123456789012"
    assert res["bucket"] == "vision-test-bucket"


def test_mocked_connectivity_verification_failure(mock_s3, mock_sts):
    """
    Tests connectivity failure reporting when S3 head_bucket returns 403 Forbidden.
    """
    error_response = {"Error": {"Code": "403", "Message": "Forbidden"}}
    mock_s3.head_bucket.side_effect = ClientError(error_response, "HeadBucket")

    manager = AWSVisionAuditManager(
        storage_mode="aws_s3",
        bucket_name="private-unreachable-bucket",
        s3_client=mock_s3,
        sts_client=mock_sts
    )
    res = manager.verify_connectivity()
    assert res["success"] is False
    assert res["status"] == "CONNECTIVITY_FAILED"
    assert "Forbidden" in res["error"]


# =====================================================================
# 7. CLOUD BATCH PIPELINE & DISPATCHER TESTS
# =====================================================================

def test_cloud_processing_event_serialization():
    """
    Tests CloudProcessingEvent creation, dict export, and JSON roundtrip.
    """
    event = CloudProcessingEvent(
        event_id="job-abc12345",
        timestamp_utc="2026-10-18T12:00:00Z",
        video_s3_uri="s3://vision-bucket/raw/test.mp4",
        output_s3_prefix="phantomroad/jobs/job-abc12345/",
        camera_id="DASHCAM_REAR_02"
    )

    d = event.to_dict()
    assert d["event_id"] == "job-abc12345"
    assert d["camera_id"] == "DASHCAM_REAR_02"

    raw_json = event.to_json()
    parsed = CloudProcessingEvent.from_dict(json.loads(raw_json))
    assert parsed.event_id == event.event_id
    assert parsed.video_s3_uri == event.video_s3_uri


def test_cloud_batch_job_dispatcher_dry_run():
    """
    Verifies that CloudBatchJobDispatcher runs safely in dry-run mode with 0 AWS calls.
    """
    manager = AWSVisionAuditManager(storage_mode="local")
    dispatcher = CloudBatchJobDispatcher(manager)

    event = dispatcher.create_job_event("s3://bucket/test.mp4")
    receipt = dispatcher.dispatch_job(event, dry_run=True)

    assert receipt["status"] == "DISPATCH_SIMULATED"
    assert receipt["mode"] == "DRY_RUN_OFFLINE"
    assert "No AWS compute was invoked" in receipt["message"]
    assert receipt["job_id"] == event.event_id


def test_cloud_batch_job_dispatcher_live_dispatch(mock_s3):
    """
    Verifies that live job dispatch writes the job request manifest into S3.
    """
    manager = AWSVisionAuditManager(
        storage_mode="aws_s3",
        bucket_name="vision-test-bucket",
        s3_client=mock_s3
    )
    dispatcher = CloudBatchJobDispatcher(manager)

    event = dispatcher.create_job_event("s3://vision-test-bucket/incoming/dashcam.mp4")
    receipt = dispatcher.dispatch_job(event, dry_run=False)

    assert receipt["status"] == "QUEUED_IN_S3"
    assert receipt["mode"] == "AWS_CLOUD"
    assert "s3://vision-test-bucket/" in receipt["job_manifest_s3_uri"]
    mock_s3.put_object.assert_called_once()


def test_cloud_pipeline_ecs_and_lambda_templates():
    """
    Verifies generation of ECS Fargate task definition and serverless Lambda code.
    """
    task_def = CloudBatchJobDispatcher.generate_ecs_task_definition_template(
        cpu="1024",
        memory="2048"
    )
    assert task_def["requiresCompatibilities"] == ["FARGATE"]
    assert task_def["cpu"] == "1024"
    assert len(task_def["containerDefinitions"]) == 1

    lambda_code = CloudBatchJobDispatcher.generate_lambda_handler_code()
    assert "def lambda_handler(event, context):" in lambda_code
    assert "AWSVisionAuditManager" in lambda_code
