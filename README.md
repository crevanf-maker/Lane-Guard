# LaneGuard — Real-Time Road Object Detection

## Overview
LaneGuard is a real-time road perception system designed to detect vehicles and pedestrians from video streams. The system uses deep learning-based object detection to process continuous input and generate annotated outputs.

## Features
- Real-time object detection using YOLO
- Video stream processing with OpenCV
- Detection of vehicles and pedestrians
- Frame-by-frame inference with annotated output

## Tech Stack
- Python
- OpenCV
- YOLO (Ultralytics / custom implementation)
- PyTorch (if used)

## How It Works
1. Input video stream is captured using OpenCV  
2. Frames are passed into the YOLO model  
3. Model performs object detection and outputs bounding boxes  
4. Frames are annotated and displayed in real-time  

## Installation

```bash
pip install -r requirements.txt
