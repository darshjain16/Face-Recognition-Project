import os
import cv2
import numpy as np
from PIL import Image

# Create LBPH recognizer (requires opencv-contrib-python)
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"

# Function to get images and labels
def get_image_with_id(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))]
    faces = []
    ids = []

    for img_path in image_paths:
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_np = np.array(img, 'uint8')

        filename = os.path.split(img_path)[-1]
        try:
            # Extract ID from filename: format User.<ID>.<SampleNum>.jpg
            id = int(filename.split('.')[1])
        except (IndexError, ValueError):
            print(f"Skipping invalid filename format: {filename}")
            continue

        faces.append(img_np)
        ids.append(id)

        # Optional: Show image during training
        cv2.imshow("Training", img_np)
        cv2.waitKey(10)

    return np.array(ids), faces

# Get training data
ids, faces = get_image_with_id(path)

# Train the recognizer
recognizer.train(faces, ids)

# Save the trained model (FIXED path using raw string)
recognizer.save(r"C:\Users\Lenovo\Desktop\face recognition\recognizer\trainingdata.yml")

# Cleanup
cv2.destroyAllWindows()
