from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/query/', views.api_query, name='api_query'),
    path('api/ingest/', views.api_ingest, name='api_ingest'),
    path('api/stats/', views.api_stats, name='api_stats'),
    path('api/clear/', views.api_clear, name='api_clear'),
]
