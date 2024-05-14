# Deepfake Detection Django App

## Overview

This repository contains a Django web application designed to detect deepfake videos. The detection is powered by a CNN+LSTM model trained on the Celeb-DF and FaceForensics++ datasets using TensorFlow-GPU.

## Features

- **Deepfake Detection**: Upload a video to check if it is a deepfake.
- **User-Friendly Interface**: Easy-to-use web interface built with Django.
- **Model Architecture**: Combines Convolutional Neural Networks (CNN) with Long Short-Term Memory (LSTM) networks for high accuracy.
- **Datasets Used**: Celeb-DF and FaceForensics++.
- **GPU Acceleration**: Utilizes TensorFlow-GPU for faster processing.

## Demo

Watch the demo video below to see the application in action:

[![Watch the demo video](https://img.youtube.com/vi/a-QFrSz9fhg/0.jpg)](https://www.youtube.com/watch?v=a-QFrSz9fhg)

## Report

If you want to read my report, you can find it [here](link-to-your-report).

## Installation

### Prerequisites

- Python 3.9
- Django 4.2.12
- TensorFlow-GPU 2.10.1
- CUDA Toolkit
- cuDNN

### Clone the Repository

```bash
git clone https://github.com/byrm-tsn/deep_fake_detection.git
cd deep_fake_detection
