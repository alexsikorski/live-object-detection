import cv2
import numpy as np

capture = cv2.VideoCapture(0)
wh_target = 416
confidence_threshold = 0.5

class_file = 'class.names'
class_names = []

with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

model_config = 'yolov3.cfg'
model_weights = 'yolov3.weights'

network = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# use cpu for network
network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def find_objects(outputs, img):
    hT, wT, cT = img.shape
    box = []
    class_ids = []
    confidence_values = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                width, height = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int(detection[0] * wT - width/2), int(detection[1] * hT - height/2)
                box.append([x, y, width, height])
                class_ids.append(class_id)
                confidence_values.append(float(confidence))

    # tells us how many detected items are present
    print(len(box))


while True:
    success, img = capture.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (wh_target, wh_target), [0, 0, 0], 1, crop=False)
    network.setInput(blob)

    layer_names = network.getLayerNames()
    output_names = [layer_names[i[0]-1] for i in network.getUnconnectedOutLayers()]

    outputs = network.forward(output_names)

    find_objects(outputs, img)

    cv2.imshow('Camera', img)
    cv2.waitKey(1)