# Deepfake Detection Django App

![Deepfake Detection](https://example.com/deepfake-detection-banner.png)

## Overview

This repository contains a Django web application designed to detect deepfake videos. The detection is powered by a CNN+LSTM model trained on the Celeb-DF and FaceForensics++ datasets using TensorFlow-GPU.

## Features

- **Deepfake Detection**: Upload a video to check if it is a deepfake.
- **User-Friendly Interface**: Easy-to-use web interface built with Django.
- **Model Architecture**: Combines Convolutional Neural Networks (CNN) with Long Short-Term Memory (LSTM) networks for high accuracy.
- **Datasets Used**: Celeb-DF and FaceForensics++.
- **GPU Acceleration**: Utilizes TensorFlow-GPU for faster processing.

## Installation

### Prerequisites

- Python 3.7+
- Django 3.2+
- TensorFlow-GPU 2.x
- CUDA Toolkit 10.1+
- cuDNN 7.6+

### Clone the Repository

```bash
git clone https://github.com/yourusername/deepfake-detection-django-app.git
cd deepfake-detection-django-app
