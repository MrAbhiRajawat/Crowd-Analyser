
import cv2
import face_recognition
import numpy as np

print("Testing Face Recognition...")

# Create a dummy image (black)
img = np.zeros((100, 100, 3), dtype=np.uint8)
print(f"Image shape: {img.shape}, dtype: {img.dtype}")

try:
    print("Trying detection on dummy image...")
    locs = face_recognition.face_locations(img)
    print(f"Success! Locs: {locs}")
except Exception as e:
    print(f"FAILED on dummy: {e}")

# Try with non-contiguous
print("\nTesting non-contiguous...")
img_nc = img[:, :, ::-1]
try:
    locs = face_recognition.face_locations(img_nc)
    print(f"Success on non-contiguous! Locs: {locs}")
except Exception as e:
    print(f"FAILED on non-contiguous: {e}")

# Try with explicit RGB
print("\nTesting explicit RGB conversion...")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
try:
    locs = face_recognition.face_locations(rgb)
    print(f"Success on RGB! Locs: {locs}")
except Exception as e:
    print(f"FAILED on RGB: {e}")

print("\nTesting face_encodings directly...")
try:
    # We pass the image and a known location (covering the whole image)
    # Location format: (top, right, bottom, left)
    loc = [(0, 100, 100, 0)]
    encs = face_recognition.face_encodings(img, loc)
    print(f"Success on encodings! Encodings found: {len(encs)}")
except Exception as e:
    print(f"FAILED on encodings: {e}")

