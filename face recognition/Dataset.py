import cv2
import numpy as np
import sqlite3
import os

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cam = cv2.VideoCapture(0)

# Ensure database and table exist
def initialize_database():
    conn = sqlite3.connect("FaceBase.db")  # Unified name
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS People (
            ID INTEGER PRIMARY KEY,
            Name TEXT,
            Age INTEGER,
            Gender TEXT,
            CR TEXT
        );
    """)
    conn.commit()
    conn.close()

# Insert or update user record
def insert_or_update(Id, Name, Age, Gen, CR):
    conn = sqlite3.connect("FaceBase.db")  # Same name as above
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM People WHERE ID=?", (Id,))
    is_record_exist = cursor.fetchone()

    if is_record_exist:
        cursor.execute("UPDATE People SET Name=?, Age=?, Gender=?, CR=? WHERE ID=?", (Name, Age, Gen, CR, Id))
    else:
        cursor.execute("INSERT INTO People (ID, Name, Age, Gender, CR) VALUES (?, ?, ?, ?, ?)", (Id, Name, Age, Gen, CR))

    conn.commit()
    conn.close()

# Create dataset directory if it doesn't exist
if not os.path.exists("dataSet"):
    os.makedirs("dataSet")

# Initialize DB
initialize_database()

# Take user input
try:
    Id = int(input('Enter User Id: '))
    name = input('Enter User Name: ')
    age = int(input('Enter User Age: '))
    gen = input('Enter User Gender: ')
    cr = input('Enter User Criminal Records: ')
except ValueError:
    print("Invalid input. ID and Age must be numbers.")
    cam.release()
    cv2.destroyAllWindows()
    exit()

# Insert or update DB entry
insert_or_update(Id, name, age, gen, cr)

# Capture face dataset
sample_num = 0
while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        sample_num += 1
        face_img = gray[y:y+h, x:x+w]
        filename = f"dataSet/User.{Id}.{sample_num}.jpg"
        cv2.imwrite(filename, face_img)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Capturing Face", img)
        cv2.waitKey(100)

    if sample_num >= 20:
        print("Dataset collection completed.")
        break

# Release camera and destroy windows
cam.release()
cv2.destroyAllWindows()
