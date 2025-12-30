import sys
print("Starting imports...")
try:
    import cv2
    print("cv2 imported")
    import torch
    print("torch imported")
    import deepface
    print("deepface imported")
except Exception as e:
    print(f"Error: {e}")
print("Imports done.")
