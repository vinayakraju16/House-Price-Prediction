from django.urls import path

from .views import harris_schema, health, model_info, monitoring, predict, predict_harris, readiness

urlpatterns = [
   path('health/', health, name='health'),
   path('ready/', readiness, name='ready'),
   path('predict/', predict, name='predict'),
   path('model-info/', model_info, name='model-info'),
   path('harris-schema/', harris_schema, name='harris-schema'),
   path('predict-harris/', predict_harris, name='predict-harris'),
   path('monitoring/', monitoring, name='monitoring'),
]
