import uuid

from django.db import migrations, models


def populate_request_ids(apps, schema_editor):
    prediction_log = apps.get_model("api", "PredictionLog")
    for primary_key in prediction_log.objects.values_list("pk", flat=True):
        prediction_log.objects.filter(pk=primary_key).update(request_id=uuid.uuid4())


class Migration(migrations.Migration):
    dependencies = [("api", "0001_predictionlog")]

    operations = [
        migrations.AddField(
            model_name="predictionlog",
            name="request_id",
            field=models.UUIDField(editable=False, null=True),
        ),
        migrations.AddField(
            model_name="predictionlog",
            name="endpoint",
            field=models.CharField(db_index=True, default="predict", max_length=64),
        ),
        migrations.AddField(
            model_name="predictionlog",
            name="status",
            field=models.CharField(db_index=True, default="success", max_length=16),
        ),
        migrations.AddField(
            model_name="predictionlog",
            name="error_code",
            field=models.CharField(blank=True, max_length=64),
        ),
        migrations.AddField(
            model_name="predictionlog",
            name="latency_ms",
            field=models.FloatField(default=0),
        ),
        migrations.AlterField(
            model_name="predictionlog",
            name="input_features",
            field=models.JSONField(default=dict),
        ),
        migrations.AlterField(
            model_name="predictionlog",
            name="prediction",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="predictionlog",
            name="range_low",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="predictionlog",
            name="range_high",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="predictionlog",
            name="confidence_level",
            field=models.CharField(blank=True, max_length=16),
        ),
        migrations.RunPython(
            populate_request_ids,
            migrations.RunPython.noop,
        ),
        migrations.AlterField(
            model_name="predictionlog",
            name="request_id",
            field=models.UUIDField(default=uuid.uuid4, editable=False, unique=True),
        ),
    ]
