import uuid

from django.db import models


class PredictionLog(models.Model):
    """A privacy-minimal inference event used for operational monitoring."""

    request_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    endpoint = models.CharField(max_length=64, default="predict", db_index=True)
    status = models.CharField(max_length=16, default="success", db_index=True)
    error_code = models.CharField(max_length=64, blank=True)
    latency_ms = models.FloatField(default=0)
    input_features = models.JSONField(default=dict)
    prediction = models.FloatField(null=True, blank=True)
    range_low = models.FloatField(null=True, blank=True)
    range_high = models.FloatField(null=True, blank=True)
    confidence_level = models.CharField(max_length=16, blank=True)
    model_version = models.CharField(max_length=100, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.endpoint}:{self.status}:{self.request_id}"
