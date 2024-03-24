import mediapipe as mp
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
import time
import math

height = 0
stable_height = 0
landmark_data = {}
shoulder_distance_inch = 0
hip_distance_inch = 0
size = ""

def height_cal():
    global stable_height, height, landmark_data
    confidence = 0
    d = 0
    stable_height=0
    proto = "MobileNetSSD_deploy.prototxt.txt"
    mobile_ssd = "MobileNetSSD_deploy.caffemodel"

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    COLORS = (0,255,0)

    confidence_threshold = 0.5

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(proto,mobile_ssd)

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = FaceMeshDetector(maxFaces=1)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame, faces = detector.findFaceMesh(frame, draw=False)

            if faces:
                face = faces[0]
                pointLeft = face[145]
                pointRight = face[374]

                # Drawing
                w , _= detector.findDistance(pointLeft, pointRight)
                W = 6.3

                # # Finding the Focal Length
                #d = 50
                #f = (w*d)/W
                #print(f)

                # Finding distance

                f = 840
                d = (W * f) / w
                # print(int(d))

                if int(d) == 240:
                    tic = time.time()
                    #get frame size, h = 720, w = 1080
                    (h, w) = frame.shape[:2]

                    #input blob for the image by resizing to a fixed 300x300 pixels and then normalizing it
                    # (note: normalization is done via the authors of the MobileNet SSD implementation)
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, size=(300,300), mean=127.5)

                    # pass the blob through the network and obtain the detections and predictions
                    print("[INFO] computing object detections...")
                    net.setInput(blob)
                    detections = net.forward()# size is 1*1*N*7 [0,class_index,probability,x1, y1, x2, y2]

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
                        # real height
                        height = (stable_height* d)/f

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS, 2)
                        cv2.putText(frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS, 2)

                        # Get pose landmarks
                        # Recolor Feed
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Make Detections
                        results = holistic.process(image)
                        # Recolor image back to BGR for rendering
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        if results.pose_landmarks:
                            for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
                                landmark_x = landmark.x
                                landmark_y = landmark.y
                                landmark_data[landmark_id] = [landmark_x, landmark_y]
                                print(f"Landmark {landmark_id}: ({landmark_x}, {landmark_y})")

                        measurements_cal()
                cvzone.putTextRect(frame, f'Face Detected',
                                (face[10][0] - 100, face[10][1] - 50),
                                scale=2)
            # # Get pose landmarks
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # # Make Detections
            results = holistic.process(image)
            # # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # drawing Right hand

            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )
            #  drawing left hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )
            # pose detection
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )

            _, buffer = cv2.imencode('.jpg', image)
            data = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
            # cv2.imshow('Realtime body measurements', image)

            # print("height",height)
            label = "person:%.2f"%(confidence * 100.0)

            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break

    cap.release()
    # cv2.destroyAllWindows()

def get_height():
    global height
    return height

def convert_normalized_to_pixel(normalized_coordinate, pixel_height):
    return normalized_coordinate * pixel_height

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def measurements_cal():
    global height, stable_height, landmark_data, shoulder_distance_inch, hip_distance_inch, size
    data = landmark_data

    # Normalized coordinates
    shoulder1_normalized = data[11]#(0.4713554382324219, 0.3730059862136841)
    shoulder2_normalized = data[12]#(0.37547117471694946, 0.37117284536361694)
    hip1_normalized = data[23]#(0.4713554382324219, 0.3730059862136841)
    hip2_normalized = data[24]#(0.37547117471694946, 0.37117284536361694)

    # Real height and pixel height
    real_height_cm = height #159.34002292963467
    pixel_height = stable_height #557

    # Convert normalized coordinates to pixel coordinates
    shoulder1_pixel = (convert_normalized_to_pixel(shoulder1_normalized[0], pixel_height),
                    convert_normalized_to_pixel(shoulder1_normalized[1], pixel_height))
    shoulder2_pixel = (convert_normalized_to_pixel(shoulder2_normalized[0], pixel_height),
                    convert_normalized_to_pixel(shoulder2_normalized[1], pixel_height))
    hip1_pixel = (convert_normalized_to_pixel(hip1_normalized[0], pixel_height),
                    convert_normalized_to_pixel(hip1_normalized[1], pixel_height))
    hip2_pixel = (convert_normalized_to_pixel(hip2_normalized[0], pixel_height),
                    convert_normalized_to_pixel(hip2_normalized[1], pixel_height))


    # Calculate distance in pixels
    shoulder_distance_pixels = calculate_distance(shoulder1_pixel , shoulder2_pixel)
    hip_distance_pixels = calculate_distance(hip1_pixel , hip2_pixel)

    # Convert distance to inch
    shoulder_distance_inch = (real_height_cm / pixel_height) * shoulder_distance_pixels
    hip_distance_inch = (real_height_cm / pixel_height) * hip_distance_pixels * 2 *2

    if shoulder_distance_inch >= 21.00:
        size="XXXl"
    elif shoulder_distance_inch>= 20.00:
        size = "XXL"
    elif shoulder_distance_inch >= 19.00:
        size = "XL"
    elif shoulder_distance_inch >= 18.00:
        size = "L"
    elif shoulder_distance_inch >= 17.00:
        size = "M"
    else:
        size = "S"

def get_shoulder():
    global shoulder_distance_inch
    return shoulder_distance_inch
def get_hip():
    global hip_distance_inch
    return hip_distance_inch
def get_size():
    global size
    return size
