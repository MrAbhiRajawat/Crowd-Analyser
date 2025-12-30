# Merged Security Backend

This system combines **Face Recognition** and **Object/Action Detection**.

## Setup Instructions

1.  **Reference Images**:
    *   Open the folder `Target_Images` in this directory.
    *   Paste approximately 15 images of the person you want to recognize.
    *   The system uses these images to learn the face, making it robust to makeup, beards, and lighting changes.

2.  **Dependencies**:
    Ensure you have the following installed:
    ```bash
    pip install opencv-python face_recognition torch torchvision numpy pandas
    pip install ultralytics seaborn tqdm psutil # YOLOv5 requirements
    ```
    *Note: `face_recognition` requires `dlib`. On Windows, this can be tricky. If `pip install face_recognition` fails, you may need to install `dlib` manually or use a pre-built wheel.*

3.  **Running the System**:
    Run the script:
    ```bash
    python merged_backend.py
    ```

## Features
*   **Target Face Detection**: Recognizes the person from the `Target_Images` folder. Use multiple images for better accuracy.
*   **Object/Action Detection**: Detects people, bags, and potential threats using YOLOv5.
*   **Robustness**: Uses Deep Learning embeddings for face recognition (dlib), which is very resilient to facial changes.
