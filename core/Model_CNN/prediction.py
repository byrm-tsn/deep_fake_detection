import cv2
import os
import numpy as np 
from keras.models import load_model 
from mtcnn.mtcnn import MTCNN 
from django.conf import settings

base_dir = settings.BASE_DIR

# Function to extract and crop face from a frame using detection details
def extract_and_crop_face(frame, detection, target_size=(224, 224)):
    x, y, width, height = detection['box']
    cropped_face = frame[y:y+height, x:x+width]
    return cv2.resize(cropped_face, target_size)

# Function to extract frames from a video using MTCNN for face detection
def extract_frames_with_mtcnn(video_path, max_frames=100):
    vidcap = cv2.VideoCapture(video_path)
    detector = MTCNN()
    frames = []

    while len(frames) < max_frames:
        success, frame = vidcap.read()
        if not success:
            break

        detections = detector.detect_faces(frame)
        if len(detections) == 1:
            cropped_face = extract_and_crop_face(frame, detections[0])
            frames.append(cropped_face)

    vidcap.release()
    return np.array(frames) / 255.0

# Function to load a pre-trained model
def load_trained_model(model_path):
    return load_model(model_path)

# Function to predict if a video is real or fake
def predict_video(model, video_frames):
    video_frames = np.expand_dims(video_frames, axis=0)  # Add batch dimension
    prediction = model.predict(video_frames, verbose=0)  # Get prediction
    percentage = prediction[0][0] * 100  # Convert prediction to percentage
    
    if prediction[0][0] < 0.5:
        return f"Fake ({100 - percentage:.2f}%)"
    else:
        return f"Real ({percentage:.2f}%)"

# Main function to handle the entire prediction process
def main(path):
    # Load the trained model
    model_path = os.path.join(base_dir, 'core', 'Model_CNN', '0_80_resnet50v2.h5')
    print('########################')
    print('########################')
    print(model_path)
    print('########################')
    print('########################')

    model = load_trained_model(model_path)  # Load model
    print('--------------------------')

    video_path = path  # Get the video path
    print(video_path)
    print('--------------------------')

    # Extract frames from the video
    video_frames = extract_frames_with_mtcnn(video_path, max_frames=100)
    print('--------------------------')
    print('issue arise after this')
    # Predict if the video is fake or real
    prediction = predict_video(model, video_frames)

    print("Prediction:", prediction)
    return prediction

# Wrapper function for the main process
def calculation(path):
    x = main(path)
    if "Real" in x:
        final_prediction = 'real'
    elif "Fake" in x:
        final_prediction = 'fake'
    return final_prediction 
