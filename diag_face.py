
import face_recognition
import cv2
import numpy as np
import os

print("Diagnostic Test...")

# 1. Create dummy image
img = np.zeros((300, 300, 3), dtype=np.uint8)
# Draw a white box (fake face?) - Face recognition won't find a face but face_locations might process it without crashing if format is right behavior
cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)

# 2. Save to disk
cv2.imwrite("temp_diag.jpg", img)
print("Saved temp_diag.jpg")

# 3. Load with library
try:
    print("Loading from disk via face_recognition...")
    loaded_img = face_recognition.load_image_file("temp_diag.jpg")
    print(f"Loaded shape: {loaded_img.shape}, dtype: {loaded_img.dtype}")
    
    print("Attempting location detection...")
    locs = face_recognition.face_locations(loaded_img)
    print(f"Locations found: {locs} (Empty is OK, Crash is NOT)")
    
except Exception as e:
    print(f"CRASHED loading/processing from disk: {e}")

# 4. Cleanup
if os.path.exists("temp_diag.jpg"):
    os.remove("temp_diag.jpg")
