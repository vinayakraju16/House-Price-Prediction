from django.urls import path
from .views import harris_schema, model_info, predict, predict_harris

urlpatterns = [
   path('predict/', predict, name='predict'),
   path('model-info/', model_info, name='model-info'),
   path('harris-schema/', harris_schema, name='harris-schema'),
   path('predict-harris/', predict_harris, name='predict-harris'),
]
