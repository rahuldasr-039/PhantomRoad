"""
Cloud architecture and optional AWS S3 integration module for PhantomRoad.
"""

from .aws_integration import AWSVisionAuditManager

__all__ = ["AWSVisionAuditManager"]
