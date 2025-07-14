import os
import cv2
import sqlite3
import numpy as np

# Path to Haar Cascade
face_cascade_path = r'C:\Users\Lenovo\Desktop\face recognition\haarcascade_frontalface_default.xml'

if not os.path.exists(face_cascade_path):
    print("Error: haarcascade_frontalface_default.xml not found!")
    exit()

facedetect = cv2.CascadeClassifier(face_cascade_path)
if facedetect.empty():
    print("Error: Failed to load face cascade classifier!")
    exit()

# Path to training data
training_data_path = r'C:\Users\Lenovo\Desktop\face recognition\recognizer\trainingdata.yml'
if not os.path.exists(training_data_path):
    print("Error: trainingData.yml not found!")
    exit()

# Initialize recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read(training_data_path)
except cv2.error as e:
    print(f"Error loading training data: {e}")
    exit()

# Function to get profile from DB
def getprofile(id):
    try:
        conn = sqlite3.connect("FaceBase.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM People WHERE ID=?", (id,))  # Table is 'People', column is 'ID'
        profile = cursor.fetchone()
        conn.close()
        return profile
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None

# Start camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open camera!")
    exit()

while True:
    ret, img = cam.read()
    if not ret:
        print("Error: Failed to capture image!")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_region = gray[y:y + h, x:x + w]

        if face_region.size == 0:
            print("Warning: Empty face region detected.")
            continue

        try:
            # Resize face for recognition
            face_resized = cv2.resize(face_region, (100, 100))
            id, conf = recognizer.predict(face_resized)

            print(f"Prediction ID: {id}, Confidence: {conf:.2f}")

            if conf > 60:  # Higher confidence = better match
                profile = getprofile(id)
                if profile:
                    color = (0, 255, 0)  # Green for valid match
                    cv2.putText(img, f"Name: {profile[1]}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(img, f"Age: {profile[0]}", (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    print(f"Face matched with: {profile[1]}")
                else:
                    print(f"No profile found for ID: {id}")
            else:
                print("Face not confidently matched.")

        except Exception as e:
            print(f"Recognition error: {e}")

    # Display number of faces detected
    cv2.putText(img, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Camera Feed", img)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
