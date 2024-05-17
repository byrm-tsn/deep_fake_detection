# Deepfake Detection Django App

## Overview

This repository contains a Django web application designed to detect deepfake videos. The detection is powered by a CNN+LSTM model trained on the Celeb-DF and FaceForensics++ datasets using TensorFlow-GPU.

<img width="1800" alt="ss" src="https://github.com/byrm-tsn/deep_fake_detection/assets/57181763/65f62bed-eb16-4453-a56d-793e989294fb">

## Features

- **Deepfake Detection**: Upload a video to check if it is a deepfake.
- **User-Friendly Interface**: Easy-to-use web interface built with Django.
- **Model Architecture**: Combines Convolutional Neural Networks (CNN) with Long Short-Term Memory (LSTM) networks for high accuracy.
- **Datasets Used**: Celeb-DF and FaceForensics++.
- **GPU Acceleration**: Utilizes TensorFlow-GPU for faster processing.

## Demo

Watch the demo video below to see the application in action:

https://github.com/byrm-tsn/deep_fake_detection/assets/57181763/7dbb2058-2cf7-4d6f-a50d-ab2e282658cc

## Report

If you want to read my report, you can find it [here](https://github.com/byrm-tsn/deep_fake_detection/blob/main/Documentation/deepfake_detection_fyp_report.pdf).

## Installation

### Prerequisites

- Python 3.9
- Django 4.2.12
- TensorFlow-GPU 2.10.1
- CUDA Toolkit v2.14
- cuDNN v9.0
### Technologies Used

[![My Skills](https://skillicons.dev/icons?i=vscode,github,django,js,html,css,git,opencv,py,tensorflow)](https://skillicons.dev)

### Clone the Repository

```bash
git clone https://github.com/byrm-tsn/deep_fake_detection.git
cd deep_fake_detection
python manage.py runserver
