# Live Object Detection
Detect objects in a webcam feed using OpenCV. 
## What's being used
- Python 3.8
- OpenCV
- [YOLO V3](https://pjreddie.com/darknet/yolo/) tiny/416 cfg and weights
- Your CPU/GPU
## How to run
- Download respective [YOLO V3 cfg and weights ](https://pjreddie.com/darknet/yolo/) files, I used YOLOv3-416 and YOLOv3-tiny.
- Configure as desired.
- Run *detect.py*.
## Configurations
More than one camera?
```python
capture = cv2.VideoCapture(0) # where 0 is the first connected device
```
How do I display objects with lower confidence percentages?
```python
confidence_threshold = 0.8 # alter this value where 0.8 is 80%
```
How do I use my gpu?
```python
# uncomment these
#network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```
## Result
![Detecting a mobile phone and a person](screenshots/cell%20phone.png)