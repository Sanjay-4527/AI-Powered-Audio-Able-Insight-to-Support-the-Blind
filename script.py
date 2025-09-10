import cv2
import numpy as np
import os
import pyttsx3
import time
import threading

# -----------------------------
# File paths (relative inside yolo/ folder)
# -----------------------------
classes_file = "yolo/coco.names.txt"
config_path = "yolo/yolov3.cfg.txt"
weights_path = "yolo/yolov3.weights"

# Load class names
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize Text-to-Speech
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# -----------------------------
# Check if YOLO files exist
# -----------------------------
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Error: {weights_path} not found!")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Error: {config_path} not found!")
if not os.path.exists(classes_file):
    raise FileNotFoundError(f"Error: {classes_file} not found!")

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# -----------------------------
# Camera parameters (calibrate if needed)
# -----------------------------
KNOWN_HEIGHT = 1.7   # Average object height in meters (example)
FOCAL_LENGTH = 600   # Focal length in pixels (adjust if needed)

# -----------------------------
# Camera setup
# -----------------------------
# Laptop webcam (default)
cap = cv2.VideoCapture(0)

# If you want to use mobile IP Webcam, comment above and uncomment below:
# cap = cv2.VideoCapture("http://192.168.xx.xx:8080/video")  # Replace with your IP

if not cap.isOpened():
    raise RuntimeError("❌ Could not open camera. Check your webcam or IP stream!")

last_spoken_time = {}
distance_buffer = []

frame = None

def read_frame():
    """Threaded function to continuously grab frames."""
    global frame
    while True:
        ret, current_frame = cap.read()
        if ret:
            frame = cv2.resize(current_frame, (320, 240))  # smaller for speed

# Start reading frames in background thread
thread = threading.Thread(target=read_frame, daemon=True)
thread.start()

try:
    while True:
        if frame is None:
            continue  # wait until frame is available

        height, width, _ = frame.shape

        # YOLO object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416),
                                     (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # detection threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if h > 0:  # distance estimation
                    distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / h
                    distance_cm = distance * 100

                    distance_buffer.append(distance_cm)
                    if len(distance_buffer) > 5:
                        distance_buffer.pop(0)
                    smoothed_distance = np.mean(distance_buffer)

                    current_time = time.time()
                    if label not in last_spoken_time or (current_time - last_spoken_time[label] >= 10):
                        speak(f"The {label} is {smoothed_distance:.1f} centimeters ahead of you")
                        last_spoken_time[label] = current_time

        cv2.imshow('YOLO Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹️ Streaming stopped by Ctrl+C")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Resources released and windows closed.")
