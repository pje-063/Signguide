#test.py
import cv2
import os
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model

def custom_load_model(model_path):
    if not os.path.exists(model_path):
        print(f" Model file not found: {model_path}")
        return None

    try:
        model = load_model(model_path, compile=False)
        print(" Model loaded successfully!")
        return model
    except Exception as e:
        print(f" Error loading model: {e}")
        return None

model_path = "Model/keras_model.h5"
labels_path = "Model/labels.txt"

model = custom_load_model(model_path)

classifier = None
if os.path.exists(labels_path):
    try:
        classifier = Classifier(model_path, labels_path)
        print(" Classifier initialized successfully!")
    except Exception as e:
        print(f" Error initializing classifier: {e}")
else:
    print(f" Labels file not found: {labels_path}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Error: Cannot open webcam.")
    exit()

detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
labels = ["1", "2", "3", "5", "6", "A", "B", "C","F","E", "I"]

while True:
    success, img = cap.read()
    if not success:
        print(" Warning: Frame capture failed. Retrying...")
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping remains within image boundaries
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size > 0:  # Ensure valid crop
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            try:
                if aspectRatio > 1:
                    # Resize height to 300 and adjust width
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))

                    # Center horizontally
                    wGap = (imgSize - wCal) // 2
                    imgWhite[:, wGap:wGap + wCal] = imgResize

                else:
                    # Resize width to 300 and adjust height
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))

                    # Center vertically
                    hGap = (imgSize - hCal) // 2
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                # Get Prediction (only if classifier is initialized)
                if classifier:
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        print(prediction, index)

                        #  Ensure index is within label range
                        predicted_label = labels[index] if index < len(labels) else "Unknown"

                        #  Define text position dynamically
                        text_x, text_y = max(10, x - offset), max(30, y - offset - 10)

                        #  Draw text with black background for visibility
                        cv2.rectangle(imgOutput, (text_x - 5, text_y - 30), (text_x + 60, text_y + 5), (0, 0, 0), -1)
                        cv2.putText(imgOutput, predicted_label, (text_x, text_y), 
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                        # Draw bounding box around detected hand
                        cv2.rectangle(imgOutput, (x - offset, y - offset), 
                                    (x + w + offset, y + h + offset), (255, 0, 255), 4)

                # Display images
                #cv2.imshow("ImageCrop", imgCrop)
                #cv2.imshow("ImageWhite", imgWhite)

            except Exception as e:
                print(f" Error processing image: {e}")

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    if key == 27:  # Press 'ESC' to exit
        print(" Exiting...")
        break

cap.release()
cv2.destroyAllWindows()