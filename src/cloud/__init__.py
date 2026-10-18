"""
Stage 4: Cloud Architecture & Optional AWS S3 Integration Module.
"""

from .aws_integration import (
    AWSVisionAuditManager,
    CloudStorageMode,
    CloudAuditBundleResult
)
from .cloud_pipeline import (
    CloudProcessingEvent,
    CloudBatchJobDispatcher
)

__all__ = [
    "AWSVisionAuditManager",
    "CloudStorageMode",
    "CloudAuditBundleResult",
    "CloudProcessingEvent",
    "CloudBatchJobDispatcher"
]
