from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    # Home page
    path('', views.IndexView.as_view(), name='index'),

    # URL for uploading videos
    path('upload_video/', views.upload_video, name='upload_video'),

    # URL for processing videos
    path('process_video/', views.process_video, name='process_video'),

    # URL for the record view
    path('record/', views.RecordView.as_view(), name='record'),

    # URL for the second record view
    path('record2/', views.SecondRecordView.as_view(), name='record2'),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
