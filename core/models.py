from django.db import models

# Create your models here.

class Video(models.Model):
    # FileField to store the uploaded video files
    file = models.FileField(upload_to='media/')
    
    # DateTimeField to store the upload date of the video
    upload_date = models.DateTimeField(auto_now_add=True)
    
    # Choices for the status of the video
    STATUS_CHOICES = [
        ('R', 'Real'),  # Real video
        ('F', 'Fake'),  # Fake video
    ]
    
    # CharField to store the status of the video with choices from STATUS_CHOICES
    status = models.CharField(max_length=1, choices=STATUS_CHOICES, default='R')

    def __str__(self):
        # String representation of the Video model
        return self.file.name
