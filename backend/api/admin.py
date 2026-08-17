from django.contrib import admin

from .models import PredictionLog


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "request_id", "endpoint", "status", "model_version", "latency_ms", "prediction")
    list_filter = ("endpoint", "status", "model_version", "confidence_level")
    search_fields = ("request_id", "model_version", "error_code")
    readonly_fields = tuple(field.name for field in PredictionLog._meta.fields)
