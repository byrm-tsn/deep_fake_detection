from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, LeakyReLU, TimeDistributed, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.applications import ResNet50, ResNet50V2, EfficientNetB0,InceptionV3, InceptionResNetV2, ResNet152V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Set mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Configure GPU settings for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
        
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# Custom Sequence generator for video frames
class VideoFrameSequenceGenerator(Sequence):
    def __init__(self, dataframe, batch_size=1, seq_length=100, frame_dim=(224, 224), shuffle=True, augment=False):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.frame_dim = frame_dim
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.dataframe))
        self.datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2], 
            fill_mode='nearest'
        )

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, idx):
        batch_data = []
        batch_labels = []

        # Select real and fake video indices
        real_indices = [i for i, label in enumerate(self.dataframe['label']) if label == 1]
        fake_indices = [i for i, label in enumerate(self.dataframe['label']) if label == 0]

        selected_real_indices = np.random.choice(real_indices, self.batch_size // 1, replace=False)
        selected_fake_indices = np.random.choice(fake_indices, self.batch_size // 1, replace=False)

        # Load and preprocess video frames
        for index in np.concatenate([selected_real_indices, selected_fake_indices]):
            row = self.dataframe.iloc[index]
            video_path = row['path']
            label = row['label']
            video_frames = self.load_video(video_path)
            if video_frames is not None:
                if video_frames.shape == (self.seq_length, *self.frame_dim):
                    if self.augment:
                        video_frames = self.augment_frames(video_frames)
                    batch_data.append(video_frames)
                    batch_labels.append(label)

        return np.array(batch_data), np.array(batch_labels)

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)
            if len(frames) == self.seq_length:
                break
        cap.release()

        if len(frames) < self.seq_length:
            # Pad the frames with zeros or repeat the last frame
            num_padding = self.seq_length - len(frames)
            padding_frames = []
            if frames:
                last_frame = frames[-1]
                padding_frames = [last_frame] * num_padding
            else:
                padding_frames = [np.zeros((self.frame_dim[0], self.frame_dim[1], 3), dtype=np.uint8)] * num_padding
            frames.extend(padding_frames)

        # Resize frames
        resized_frames = [cv2.resize(frame, (self.frame_dim[0], self.frame_dim[1])) for frame in frames]
        return np.array(resized_frames) / 255.0

    def augment_frames(self, frames):
        return np.array([self.datagen.random_transform(frame) for frame in frames])

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Build LSTM model with ResNet50V2 base
def build_lstm_model(seq_length=100, frame_dim=(224, 224, 3), lstm_units=256, num_classes=1):
    base_model = ResNet50V2(include_top=False, input_shape=frame_dim, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(TimeDistributed(base_model, input_shape=(seq_length, *frame_dim)))
    model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D(2,2)))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=False,
                             kernel_regularizer=L1(0.1),
                             recurrent_regularizer=L2(0.1))))
    model.add(Dropout(0.3))
    model.add(Dense(lstm_units // 2, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    optimiser =  Adam(learning_rate=1e-5)
    model.compile(optimizer=optimiser, loss='binary_crossentropy', metrics=['accuracy'])
    return model 

# Evaluate and visualize LSTM model performance
def evaluate_and_visualize_lstm(model, test_gen, model_name="LSTM Model"):
    y_true = []
    y_pred = []
    y_pred_proba = []  

    for test_features, test_labels in tqdm(test_gen, total=len(test_gen), desc="Evaluating"):
        predictions = model.predict_on_batch(test_features)
        y_pred_proba.extend(predictions.flatten())
        predicted_classes = (predictions > 0.5).astype(int)
        y_pred.extend(predicted_classes.flatten())
        y_true.extend(test_labels.flatten())

    # Ensure correct data types
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    # Display metrics
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))

    # Calculate and plot ROC AUC if valid
    if len(np.unique(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc:.4f}")
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    else:
        print("ROC AUC cannot be calculated with one class only.")

# Paths to data and metadata
feature_dir = 'video_data'
df = pd.read_csv('metadata.csv directory')

# Callbacks for training
checkpoint = ModelCheckpoint('lstm_model_best.h5', monitor='val_accuracy', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='min', restore_best_weights=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=0.000001)

# Data generators
train_gen = VideoFrameSequenceGenerator(df[df['set'] == 'train'], batch_size=1, seq_length=100, frame_dim=(224, 224, 3), augment=False)
val_gen = VideoFrameSequenceGenerator(df[df['set'] == 'validation'], batch_size=1, seq_length=100, frame_dim=(224, 224, 3), augment=False)
test_gen = VideoFrameSequenceGenerator(df[df['set'] == 'test'], batch_size=1, seq_length=100, frame_dim=(224, 224, 3), augment=False, shuffle=False)

# Build and compile the LSTM model
lstm_model = build_lstm_model()

# Train the model
history = lstm_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Plot training history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Load the best model weights
lstm_model.load_weights('lstm_model_best.h5')   

# Evaluate the model on the test set
roc_auc = evaluate_and_visualize_lstm(lstm_model, test_gen)
print(f"ROC AUC: {roc_auc}")
