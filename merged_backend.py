
import cv2
import os
import torch
import numpy as np
from deepface import DeepFace
import threading
import time
from twilio.rest import Client # Twilio Import

# --- CONFIGURATION ---
TARGET_IMAGES_DIR = "Target_Images"
MODEL_CONFIDENCE = 0.4
FACE_SIMILARITY_THRESHOLD = 0.4 
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# --- TWILIO CREDENTIALS (PLACEHOLDERS) ---
TWILIO_SID = "ACbb77381a4960ee5aa636ac60dfd0d089"
TWILIO_AUTH_TOKEN = "ba61117fb64b44cbe8d05887ad4864c8"
TWILIO_FROM_NUMBER = "+16076008989"
TWILIO_TO_NUMBER = "+919001403069"

# Alert Cooldown (in seconds) so we don't spam SMS
SMS_COOLDOWN = 30 

class SecurityBackend:
    def __init__(self):
        print("Initializing Security Backend with DeepFace...")
        
        # Twilio Setup
        self.twilio_client = None
        try:
            if "YOUR_" not in TWILIO_SID: # Only init if user filled it
                 self.twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
                 print("Twilio Client Initialized.")
            else:
                 print("Twilio: Credentials not set. SMS will start printing to console instead.")
        except Exception as e:
            print(f"Twilio Init Error: {e}")

        self.last_sms_time = 0
        self.suspicious_frame_counter = 0 # To track persistence

        # 1. Initialize Face Recognition (rest of init...)
        self.known_embeddings = []
        self.face_rec_enabled = True
        self.face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        
        # DeepFace needs a dummy call to load weights initially
        try:
            # Create a black image
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            # This triggers model download/load
            DeepFace.represent(img_path=dummy_img, model_name="VGG-Face", enforce_detection=False)
            print("DeepFace Model Loaded.")
        except Exception as e:
            # Safe print for unicode errors
            print("DeepFace Init Warning: Could not download/load model on first try. Check connection.")

        self.load_reference_faces()
        
        # 2. Initialize Object Detection (YOLOv5)
        print("Loading YOLOv5 Model...")
        try:
            self.output_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.output_model.classes = None 
            print("YOLOv5 Loaded.")
        except Exception as e:
            print(f"ERROR: YOLOv5 failed to load: {e}")
            self.output_model = None
            
        # 3. Optical Flow state
        self.prev_gray = None

        # --- Control Flags ---
        self.show_crowd = True
        self.show_suspicious = True
        self.show_face = True

    def send_twilio_alert(self, body_text):
        current_time = time.time()
        if current_time - self.last_sms_time < SMS_COOLDOWN:
            return # Skip if too soon

        print(f"\n[SENDING SMS] {body_text}\n")
        self.last_sms_time = current_time
        
        if self.twilio_client:
            try:
                message = self.twilio_client.messages.create(
                    body=body_text,
                    from_=TWILIO_FROM_NUMBER,
                    to=TWILIO_TO_NUMBER
                )
                print(f"Twilio Message Sent: {message.sid}")
            except Exception as e:
                print(f"Twilio Send Failed: {e}")

    def load_reference_faces(self):
        # ... (Same as before)
        print(f"Loading reference faces from {TARGET_IMAGES_DIR}...")
        if not os.path.exists(TARGET_IMAGES_DIR):
            os.makedirs(TARGET_IMAGES_DIR)
            return

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(TARGET_IMAGES_DIR) if f.lower().endswith(valid_extensions)]
        
        count = 0
        for filename in image_files:
            filepath = os.path.join(TARGET_IMAGES_DIR, filename)
            try:
                embedding = DeepFace.represent(img_path=filepath, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
                self.known_embeddings.append(embedding)
                count += 1
            except Exception:
                pass
        
        print(f"Loaded {count} reference voices/faces.")

    # ... check_face_match remains same ...
    def check_face_match(self, face_crop):
        if not self.known_embeddings: return False
        try:
            target_embedding = DeepFace.represent(img_path=face_crop, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
            for known_emb in self.known_embeddings:
                a = np.array(target_embedding)
                b = np.array(known_emb)
                cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                if (1 - cos_sim) < FACE_SIMILARITY_THRESHOLD: return True
        except: pass
        return False

    def process_frame(self, frame):
        crowd_count = 0
        weapon_detected = False
        
        # --- 1. Face Detection & Recognition ---
        if self.show_face:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                label = "Unknown"
                if len(self.known_embeddings) > 0:
                     face_crop = frame[y:y+h, x:x+w]
                     if face_crop.size > 0 and self.check_face_match(face_crop):
                         label = "TARGET MATCH"
                         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # --- 2. Object Detection (YOLOv5) & Crowd Counting ---
        if self.output_model:
            try:
                if self.show_crowd or self.show_suspicious:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.output_model(img_rgb)
                    detections = results.xyxy[0].cpu().numpy()
                    
                    for det in detections:
                        x1, y1, x2, y2, conf, cls = det
                        if conf > MODEL_CONFIDENCE:
                            c_name = self.output_model.names[int(cls)]
                            
                            if c_name == 'person': 
                                crowd_count += 1
                            
                            if self.show_suspicious:
                                if c_name in ['knife', 'scissors', 'baseball bat']:
                                    weapon_detected = True
                                    label_yolo = f"WEAPON: {c_name} {conf:.2f}"
                                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                                    cv2.putText(frame, label_yolo, (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                elif c_name != 'person':
                                    label_yolo = f"{c_name} {conf:.2f}"
                                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 100, 0), 2)
                                    cv2.putText(frame, label_yolo, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except Exception:
                pass
        
        # --- 3. Optical Flow (Violent Motion) ---
        motion_level = 0
        VIOLENT_THRESHOLD = 25.0 
        
        if self.show_suspicious:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is None:
                self.prev_gray = gray
            else:
                flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                top_motion = np.percentile(mag, 98)
                motion_level = top_motion
                self.prev_gray = gray
        
        if self.show_suspicious:
             cv2.putText(frame, f"Motion Level: {motion_level:.1f} / {VIOLENT_THRESHOLD}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --- SMS ALERT LOGIC ---
        is_violent = (motion_level > VIOLENT_THRESHOLD)
        
        # 1. Suspicious Activity Persistance Check (4 frames)
        if weapon_detected or is_violent:
             self.suspicious_frame_counter += 1
        else:
             self.suspicious_frame_counter = 0 # Reset if chain breaks

        if self.suspicious_frame_counter >= 4:
             # Trigger SOS
             cause = "WEAPON" if weapon_detected else "VIOLENT FIGHT"
             self.send_twilio_alert(f"SOS: Suspicious Activity Detected! Cause: {cause}")
             self.alert_timer = 20 # Show on screen alert longer

        # 2. Crowd Overload Check
        if crowd_count > 8:
            self.send_twilio_alert(f"ALERT: Overcrowding Detected! Count: {crowd_count}")


        # --- Visual Alert Persistence ---
        if self.suspicious_frame_counter > 0: # If currently suspicious
            self.alert_timer = 10 
        
        if self.show_suspicious and hasattr(self, 'alert_timer') and self.alert_timer > 0:
            self.alert_timer -= 1
            if weapon_detected: alert_text = "ALERT: WEAPON DETECTED"
            elif is_violent: alert_text = "ALERT: VIOLENT MOTION (FIGHT) DETECTED"
            else: alert_text = "ALERT: SUSPICIOUS ACTIVITY"
            cv2.putText(frame, alert_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)
        
        if self.show_crowd:
            color = (0, 255, 255)
            if crowd_count > 8: color = (0, 0, 255) # Red if overcrowded
            cv2.putText(frame, f"Crowd Count: {crowd_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
        return frame

def start_backend():
    backend = SecurityBackend()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Webcam not found.")
        return

    print("Backend Running with DeepFace. Press 'q' to quit.")
    print(f"\n[IMPORTANT] Put target images in: {os.path.abspath(TARGET_IMAGES_DIR)}\n")
    print("CONTROLS: [C] Crowd | [S] Suspicious | [F] Face | [K] All | [Q] Exit")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        output_frame = backend.process_frame(frame)
        
        # Draw status of toggles
        status_text = f"C:{'ON' if backend.show_crowd else 'OFF'} S:{'ON' if backend.show_suspicious else 'OFF'} F:{'ON' if backend.show_face else 'OFF'}"
        cv2.putText(output_frame, status_text, (10, output_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Unified Security System (DeepFace)', output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            backend.show_crowd = not backend.show_crowd
        elif key == ord('s'):
            backend.show_suspicious = not backend.show_suspicious
        elif key == ord('f'):
            backend.show_face = not backend.show_face
        elif key == ord('k'):
            backend.show_crowd = True
            backend.show_suspicious = True
            backend.show_face = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_backend()

