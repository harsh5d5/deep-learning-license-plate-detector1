# Precision Real-Time ANPR
### Developed by: Harsh Jayeshbhai Patel 

A high-performance Automatic Number Plate Recognition (ANPR) system using YOLOv8, SORT tracking, and EasyOCR with persistent relative coordinate locking.

## Features 
- **Real-Time 1080p Processing**: Optimized for CPU speed.
- **Relative Plate Locking**: Red box "sticks" to the car even if the plate is temporarily hidden.
- **Proportional Scaling**: Intelligent box resizing as cars approach the camera.
- **Memory Management**: Automatic "Exit Zone" cleanup at the bottom of the frame.
- **Selective OCR**: Skips slow processing once confidence is >90%.

## Key Libraries (Versions)
- **ultralytics**: 8.4.21 (YOLOv8)
- **easyocr**: 1.7.2 (Text Recognition)
- **opencv-python**: 4.13.0
- **numpy**: 2.4.3
- **filterpy**: 1.4.5 (Kalman Filtering)
- **pandas**: 3.0.1 (Result Export)

## Running the Project
```bash
python main.py
```
