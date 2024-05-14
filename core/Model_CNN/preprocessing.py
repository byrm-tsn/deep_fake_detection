import os
import cv2
import hashlib
import csv
import glob
import numpy as np
from mtcnn.mtcnn import MTCNN
import concurrent.futures

# Ensure directory exists, create if not
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(f"Directory ensured: {directory}")

# Extract and crop face from a frame using detection details
def extract_and_crop_face(frame, detection, target_size=(224, 224)):
    x, y, width, height = detection['box']
    cropped_face = frame[y:y+height, x:x+width]
    return cv2.resize(cropped_face, target_size)

# Extract frames from a video using MTCNN for face detection
def extract_frames_with_mtcnn(video_path, out_path, max_frames=100):
    print(f"Extracting frames from {video_path}")
    vidcap = cv2.VideoCapture(video_path)
    detector = MTCNN()
    frames_extracted = 0

    while frames_extracted < max_frames:
        success, frame = vidcap.read()
        if not success:
            break

        detections = detector.detect_faces(frame)
        if len(detections) == 1:
            cropped_face = extract_and_crop_face(frame, detections[0])
            frame_path = os.path.join(out_path, f"frame_{frames_extracted}.jpg")
            cv2.imwrite(frame_path, cropped_face)
            frames_extracted += 1

    vidcap.release()
    print(f"Extracted {frames_extracted} frames from {video_path}")
    return frames_extracted

# Process a video: extract frames, crop faces, save to output path
def process_video(video_path, category, set_type, base_output_path):
    video_id = hashlib.md5(video_path.encode()).hexdigest()  # Unique ID based on hash
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Use video_id as the primary folder name
    out_path = os.path.join(base_output_path, set_type, category, video_id) 
    ensure_directory_exists(out_path) 

    print(f"Processing video: {video_path}")

    # Extract frames and get the number extracted
    frames_extracted = extract_frames_with_mtcnn(video_path, out_path, max_frames=100)

    return {
        "video_id": video_id,
        "category": category,
        "set_type": set_type,
        "frames_extracted": frames_extracted,
        "frame_path": out_path
    }

# List all MP4 files in a directory
def list_mp4_files(directory):
    mp4_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    return mp4_files

# Distribute videos evenly into train, validation, and test sets
def distribute_videos_evenly(fake_categories_paths, real_videos_path, max_fake_videos_per_type, max_real_videos):
    all_videos = {'train': [], 'validation': [], 'test': []}
    
    # Process fake videos
    for category_info in fake_categories_paths:
        category_videos = list_mp4_files(category_info['path'])[:max_fake_videos_per_type]
        num_train = int(0.7 * len(category_videos))
        num_val = int(0.15 * len(category_videos))
        
        # Append non-overlapping slices to the corresponding sets
        all_videos['train'].extend([(vp, 'fake') for vp in category_videos[:num_train]])
        all_videos['validation'].extend([(vp, 'fake') for vp in category_videos[num_train:num_train+num_val]])
        all_videos['test'].extend([(vp, 'fake') for vp in category_videos[num_train+num_val:]])

    # Process real videos
    real_videos = list_mp4_files(real_videos_path['path'])[:max_real_videos]
    num_train = int(0.7 * len(real_videos))
    num_val = int(0.15 * len(real_videos))
    # Append non-overlapping slices to the corresponding sets
    all_videos['train'].extend([(vp, 'real') for vp in real_videos[:num_train]])
    all_videos['validation'].extend([(vp, 'real') for vp in real_videos[num_train:num_train+num_val]])
    all_videos['test'].extend([(vp, 'real') for vp in real_videos[num_train+num_val:]])
    return all_videos

# Create video from frames
def create_video_from_frames(frames_dir, output_dir, video_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_files = sorted(
        glob.glob(os.path.join(frames_dir, '*.jpg')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
    )
    if not frame_files:
        print("No frames found in directory:", frames_dir)
        return

    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape

    video_path = os.path.join(output_dir, f'{video_name}.mp4')
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)

    video_writer.release()
    print(f'Video saved: {video_path}')

# Process directory to create videos from frames
def process_directory_for_videos(directory, output_base):
    for root, dirs, files in os.walk(directory):
        if files:  # This means we are in a directory with files
            rel_path = os.path.relpath(root, directory)
            video_name = os.path.basename(rel_path)
            output_dir = os.path.join(output_base, os.path.dirname(rel_path))
            create_video_from_frames(root, output_dir, video_name)

# Create metadata CSV file
def create_metadata_csv(dataset_path, output_csv_path):
    header = ['path', 'label', 'set', 'video_id']

    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

        # Iterate over set types (train, validation, test)
        for set_type in ['train', 'validation', 'test']:
            # Iterate over labels (fake, real)
            for label in ['fake', 'real']:
                directory_path = os.path.join(dataset_path, set_type, label)
                # Ensure the directory exists before trying to list its contents
                if not os.path.exists(directory_path):
                    continue
                # List each video in the directory
                for video_id in os.listdir(directory_path):
                    video_path = os.path.join(directory_path, video_id)
                    # Check if the path is indeed a file, assuming all files in the directory are videos
                    if os.path.isfile(video_path):
                        writer.writerow({
                            'path': video_path,
                            'label': 0 if label == 'fake' else 1,
                            'set': set_type,
                            'video_id': os.path.splitext(video_id)[0]  
                        })

def main():
    fake_categories_paths = [
        {'path': 'raw_fake_videos_dir', 'category': 'fake'}
    ]
    
    real_videos_path = {'path': 'raw_real_videos_dir', 'category': 'real'}

    max_fake_videos_per_type = 250
    max_real_videos = 250

    output_base_path = 'output dir for dataset'
    all_videos_distributed = distribute_videos_evenly(fake_categories_paths, real_videos_path, max_fake_videos_per_type, max_real_videos)

    # Assuming distribute_videos_evenly now correctly organizes videos into a dictionary of lists with tuples (video_path, category)
    metadata = []
    processed_videos = set()  # Keep track of videos already processed
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for set_type, videos in all_videos_distributed.items():
            for video_path, category in videos:
                # Check if the video has already been processed
                if video_path not in processed_videos:
                    futures.append(executor.submit(process_video, video_path, category, set_type, output_base_path))
                    processed_videos.add(video_path)  # Mark this video as processed
                
        for future in concurrent.futures.as_completed(futures):
            metadata.extend(future.result())

    # After processing videos, create videos from extracted frames
    process_directory_for_videos(output_base_path, output_base_path)

    # Create metadata CSV
    create_metadata_csv(output_base_path, os.path.join(output_base_path, "metadata.csv"))
    print("Preprocessing complete. Metadata CSV created.")

if __name__ == "__main__":
    main()
