import os
import json
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from .models import Video
from core.Model_CNN import prediction

# View for the second record page
class SecondRecordView(View):
    def get(self, request):
        videos = Video.objects.all()  # Fetch all video records from the database
        return render(request, 'record2.html', {'videos': videos})  # Render the 'record2.html' template with video data

# View for the index page
class IndexView(View):
    def get(self, request):
        return render(request, 'index2.html')  # Render the 'index2.html' template

# View for handling video uploads
@csrf_exempt
def upload_video(request):
    if request.method == 'POST':
        try:
            video_file = request.FILES['uploaded_file']  # Get the uploaded video file from the request
            video = Video(file=video_file)
            video.save()  # Save the video record in the database

            media_dir = settings.MEDIA_ROOT

            # Get the absolute path of the saved video file
            video_path = os.path.join(media_dir, video.file.name)

            return JsonResponse({'status': 'success', 'file_path': video_path})
        except Exception as e:
            print(e)
            return JsonResponse({'status': 'error', 'message': f'Error uploading the video: {str(e)}'}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

# View for processing the uploaded video
@csrf_exempt
def process_video(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            video_path = data.get('file_path')

            if not video_path:
                return JsonResponse({'status': 'error', 'message': 'File path is required'}, status=400)

            # Extract the file name from the video_path
            file_name = os.path.basename(video_path)

            # Fetch the video instance using the file name
            video = Video.objects.get(file='media/' + file_name)

            # Call the prediction function to determine if the video is real or fake
            prediction_result = prediction.calculation(video_path)

            if prediction_result == 'real':
                video.status = 'R'  
                result_message = "Video is REAL."
            else:
                video.status = 'F'  
                result_message = "Video is FAKE."

            video.save()  # Save the updated video status

            return JsonResponse({'status': 'success', 'message': result_message})
        except Video.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Video matching query does not exist.'}, status=400)
        except Exception as e:
            print(e)
            return JsonResponse({'status': 'error', 'message': f'Error processing the video: {str(e)}'}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

# View for the record page
class RecordView(View):
    def get(self, request):
        videos = Video.objects.all()  # Fetch all video records from the database
        return render(request, 'record.html', {'videos': videos})  # Render the 'record.html' template with video data
