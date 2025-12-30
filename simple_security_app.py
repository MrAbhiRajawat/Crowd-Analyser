import cv2
import os
import numpy as np
from ultralytics import YOLO
import time

# ==========================================
#        SIMPLIFIED SECURITY SYSTEM
# ==========================================

# 1. Directories
TARGET_IMAGES_DIR = os.path.join(os.getcwd(), "Target_Images")
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# 2. AI Thresholds
MODEL_CONFIDENCE = 0.25
VIOLENT_THRESHOLD = 40.0  # Motion threshold for suspicious activity

class SecurityBackend:
    def __init__(self):
        print("\n[INIT] Initializing Security System...")
        
        # --- Control Toggles (Default: ALL ON) ---
        self.show_crowd = True
        self.show_suspicious = True
        self.show_face = True
        
        # --- State Variables ---
        self.prev_gray = None
        self.suspicious_frame_counter = 0
        self.alert_timer = 0
        
        # --- Initialize Models ---
        self.init_face_detection()
        self.init_weapon_detection()

    def init_face_detection(self):
        print("[INIT] Loading Face Detection (Haar Cascade)...")
        self.face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        
        # Create target images directory if it doesn't exist
        if not os.path.exists(TARGET_IMAGES_DIR):
            print(f"[DEBUG] Creating dir: {TARGET_IMAGES_DIR}")
            os.makedirs(TARGET_IMAGES_DIR)
        
        print(f"[SUCCESS] Face Detection Active.")

    def init_weapon_detection(self):
        print("[INIT] Loading Weapon Detection (YOLOv8s)...")
        try:
            self.model = YOLO("yolov8s.pt") 
            print("[SUCCESS] YOLOv8s Weapon Detection Active.")
        except Exception as e:
            print(f"[ERROR] Failed to load YOLOv8: {e}")
            self.model = None

    def send_alert(self, text):
        # Console Alert
        print(f" >> ALARM TRIGGERED: {text}")

    def process_frame(self, frame):
        # 1. Optical Flow (Violent Motion)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_level = 0.0
        
        if self.prev_gray is not None and self.show_suspicious:
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # Use percentile to find fast moving objects (e.g. fist)
            motion_level = np.percentile(mag, 98)
        
        self.prev_gray = gray

        # 2. Main Detection Loop (YOLOv8)
        crowd_count = 0
        weapon_detected = False
        weapon_name = ""
        
        # YOLOv8 Inference
        if self.model and (self.show_crowd or self.show_suspicious):
            try:
                # Run inference (conf=0.15 for aggressive detection)
                results = self.model(frame, conf=0.15, verbose=False)
                
                for result in results:
                    for box in result.boxes:
                        # Extract Box Info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        label = self.model.names[cls]

                        # Crowd (Person is class 0 or 'person')
                        if label == 'person':
                            crowd_count += 1
                            # Draw Yellow Box for Person
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        
                        # Hidden/Suspicious Objects
                        if self.show_suspicious:
                             # List of threat objects
                             if label in ['knife', 'scissors', 'baseball bat', 'gun', 'pistol']:
                                 weapon_detected = True
                                 weapon_name = label
                                 # Draw Red Box
                                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                 cv2.putText(frame, f"WEAPON: {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                             
                             # Draw other objects (optional)
                             elif label != 'person':
                                 cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 1)
                                 cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,100,0), 1)
            except Exception as e:
                pass

        # 3. Face Detection (Always run for Crowd Counting Accuracy)
        # Relaxed Haar Cascade settings for better detection
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3)
        face_count = len(faces)
        
        # Draw Face Boxes (Only if Toggle is ON)
        if self.show_face:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 4. Logic & Alerts
        # Robust Crowd Count: Max of (YOLO People, Haar Faces)
        final_crowd_count = max(crowd_count, face_count)

        # A. Crowd Alert
        if self.show_crowd:
             color = (0, 255, 255) # Yellow
             if final_crowd_count > 7: # Crowd limit > 7
                 color = (0, 0, 255) # Red
                 self.send_alert(f"SOS: OVERCROWD WARNING! {final_crowd_count} people detected (Limit: 7)")
             
             # Display Crowd Count at Top Left
             cv2.putText(frame, f"CROWD COUNT: {final_crowd_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # B. Suspicious Alert (Weapon or Fight)
        is_violent = (motion_level > VIOLENT_THRESHOLD)
        
        # Persistence Logic 
        if weapon_detected or is_violent:
            self.suspicious_frame_counter += 1
        else:
            self.suspicious_frame_counter = 0
            
        # Trigger SOS if suspicious for more than 6 frames
        if self.suspicious_frame_counter > 6:
            cause = f"Found {weapon_name}" if weapon_detected else "Violent Motion"
            self.send_alert(f"SOS: DANGER! Suspicious Activity Detected ({cause}).")
            self.alert_timer = 20 # Keep alert on screen for ~1 sec

        # Show On-Screen Alert
        if self.suspicious_frame_counter > 0:
            self.alert_timer = 5 # Flash alert while happening
        
        if self.show_suspicious and self.alert_timer > 0:
            self.alert_timer -= 1
            alert_msg = "ALERT: SUSPICIOUS ACTIVITY"
            if weapon_detected: alert_msg = f"ALERT: WEAPON ({weapon_name})"
            elif is_violent: alert_msg = "ALERT: VIOLENT FIGHT DETECTED"
            
            # Big Red Banner
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            cv2.putText(frame, alert_msg, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Debug Motion (Bottom Left)
        if self.show_suspicious:
             cv2.putText(frame, f"Motion: {motion_level:.1f}/{VIOLENT_THRESHOLD}", (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        return frame

# ==========================================
#              MAIN LOOP
# ==========================================
def main():
    app = SecurityBackend()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] No webcam found.")
        return

    print("\n" + "="*40)
    print("   SECURITY SYSTEM (YOLOv8)  ")
    print("="*40)
    print(" [C] Toggle Crowd Counting")
    print(" [S] Toggle Suspicious Detection")
    print(" [F] Toggle Face Detection")
    print(" [K] Enable ALL Features")
    print(" [Q] Quit Application")
    print("="*40 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Process Frame
        output_frame = app.process_frame(frame)
        
        # Draw Toggles Status
        status = f"[C]:{'ON' if app.show_crowd else 'OFF'}  [S]:{'ON' if app.show_suspicious else 'OFF'}  [F]:{'ON' if app.show_face else 'OFF'}"
        cv2.putText(output_frame, status, (10, output_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Security System", output_frame)
        
        # Key Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'): app.show_crowd = not app.show_crowd
        elif key == ord('s'): app.show_suspicious = not app.show_suspicious
        elif key == ord('f'): app.show_face = not app.show_face
        elif key == ord('k'): 
            app.show_crowd = True
            app.show_suspicious = True
            app.show_face = True

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System Shutdown.")

if __name__ == "__main__":
    main()
