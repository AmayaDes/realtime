


import numpy as np
import cv2
import time
import os
import FaceDepthMeasurement


proto = "MobileNetSSD_deploy.prototxt.txt"
mobile_ssd = "MobileNetSSD_deploy.caffemodel"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = (0,255,0)

confidence_threshold = 0.5

d= FaceDepthMeasurement.d
focal = FaceDepthMeasurement.f


# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(proto,mobile_ssd)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1270)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 780)


while cap.isOpened():

    tic = time.time()
    ret,frame = cap.read()
    if not ret:
        break

    #get frame size, h = 720, w = 1080
    (h, w) = frame.shape[:2]

    #input blob for the image by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD implementation)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),0.007843,size=(300,300),mean=127.5)

    # pass the blob through the network and obtain the detections and predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward() # size is 1*1*N*7 [0,class_index,probability,x1, y1, x2, y2]

    detections = detections[0,0,:] # size is N*7 [0,class_index,probability,x1, y1, x2, y2]

    #select detecions in person and confidence_threshold > 0.2 , return detections # size is N*5 [probability,x1, y1, x2, y2]
    detections = np.asarray([det[2:] for det in detections if int(det[1]) == CLASSES.index('person') and det[2] > confidence_threshold])

    # For each detected person, find the highest and lowest points
    for det in detections:
        confidence = det[0]
        box = det[1:] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # Find the highest and lowest y-coordinates
        highest_point = y1
        lowest_point = y2

        # Calculate stable height
        stable_height = lowest_point - highest_point

        height = (stable_height* d)/focal

        print("height",height)

        # Display the stable height on the frame
        height_label = f"Stable Height: {stable_height} pixels"
        cv2.putText(frame, height_label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS, 2)

        # Draw bounding box and confidence
        label = "person:%.2f"%(confidence * 100.0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS, 2)
        cv2.putText(frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS, 2)

    toc = time.time()
    durr = float(toc-tic)

    fps = 1.0 / durr
    # cv2.putText(frame, "fps:%.3f" % fps, (20, 20), 3, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "height" % height, (20, 20), 3, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
