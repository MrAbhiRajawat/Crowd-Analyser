import cv2
import os
import numpy as np
from ultralytics import YOLO
import time
import face_recognition
from twilio.rest import Client

# ==========================================
#        FINAL SECURITY SYSTEM CONFIG
# ==========================================

# 1. Directories
TARGET_IMAGES_DIR = os.path.join(os.getcwd(), "Target_Images")
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# 2. AI Thresholds
PERSON_CONFIDENCE = 0.4  # Higher confidence for accurate person detection
WEAPON_CONFIDENCE = 0.3  # Confidence for weapon detection
VIOLENT_THRESHOLD = 40.0  # Motion threshold for suspicious activity

# 3. Twilio SMS Configuration (Replace with your details)
TWILIO_SID = os.getenv("TWILIO_SID", "YOUR_TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "YOUR_TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "YOUR_TWILIO_FROM_NUMBER")
TWILIO_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER", "YOUR_TWILIO_TO_NUMBER")
SMS_COOLDOWN = 30  # Seconds between SMS alerts

class SecurityBackend:
    def __init__(self):
        print("\n[INIT] Initializing Final Security System...")
        
        # --- Control Toggles (Default: ALL ON) ---
        self.show_crowd = True
        self.show_suspicious = True
        self.show_face = True
        
        # --- Twilio Setup ---
        self.twilio_client = None
        self.last_sms_time = 0
        try:
            if "YOUR_" not in TWILIO_SID:
                 self.twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
                 print("[INFO] Twilio SMS Service: ACTIVE")
            else:
                 print("[INFO] Twilio SMS Service: OFF (Credentials missing, using console logs)")
        except Exception as e:
            print(f"[WARN] Twilio Init Error: {e}")
        
        # --- State Variables ---
        self.prev_gray = None
        self.suspicious_frame_counter = 0
        self.alert_timer = 0
        self.known_face_encodings = []  # For SRK face recognition
        
        # --- Initialize Models ---
        self.init_face_detection()
        self.init_face_recognition()
        self.init_weapon_detection()

    def init_face_detection(self):
        print("[INIT] Loading Face Detection (Haar Cascade)...")
        self.face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        print(f"[SUCCESS] Face Detection Active.")

    def init_face_recognition(self):
        print("[INIT] Loading Face Recognition for SRK...")
        
        # Create directory if it doesn't exist
        if not os.path.exists(TARGET_IMAGES_DIR):
            print(f"[DEBUG] Creating dir: {TARGET_IMAGES_DIR}")
            os.makedirs(TARGET_IMAGES_DIR)
        
        # Load all SRK images
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(TARGET_IMAGES_DIR) if f.lower().endswith(valid_exts)]
        
        print(f"[DEBUG] Found {len(files)} images in {TARGET_IMAGES_DIR}")
        
        count = 0
        for f in files:
            try:
                path = os.path.join(TARGET_IMAGES_DIR, f)
                # Load image using OpenCV
                img = cv2.imread(path)
                if img is None:
                    print(f"[WARN] Could not read: {f}")
                    continue
                    
                # Convert BGR to RGB and ensure strictly uint8 and contiguous (CRITICAL for dlib)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb_img = np.ascontiguousarray(rgb_img, dtype=np.uint8)
                
                # Get face encoding
                encodings = face_recognition.face_encodings(rgb_img)
                if len(encodings) > 0:
                    self.known_face_encodings.append(encodings[0])
                    count += 1
                    print(f"[DEBUG] Loaded: {f}")
                else:
                    print(f"[WARN] No face found in: {f}")
            except Exception as e:
                print(f"[ERROR] Failed to load {f}: {e}")
        
        if count == 0:
            print(f"[WARN] No face encodings loaded. Face Recognition will show 'Unknown'.")
        else:
            print(f"[SUCCESS] Loaded {count} SRK face encodings for recognition.")

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
        
        # SMS Alert (With Cooldown)
        current = time.time()
        if current - self.last_sms_time > SMS_COOLDOWN:
            self.last_sms_time = current
            
            if self.twilio_client:
                try:
                    self.twilio_client.messages.create(body=text, from_=TWILIO_FROM_NUMBER, to=TWILIO_TO_NUMBER)
                    print(" >> SMS SENT Successfully.")
                except Exception as e:
                    print(f" >> SMS FAILED: {e}")
            else:
                 print(" >> (SMS Skipped - No Credentials Configured)")


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

        # 2. YOLO Person Detection (Accurate Counting)
        person_count = 0
        weapon_detected = False
        weapon_name = ""
        
        # YOLOv8 Inference with higher confidence for accurate person detection
        if self.model and self.show_crowd:
            try:
                # Run inference with higher confidence for persons
                results = self.model(frame, conf=PERSON_CONFIDENCE, verbose=False)
                
                for result in results:
                    for box in result.boxes:
                        # Extract Box Info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        label = self.model.names[cls]

                        # Count only persons with high confidence
                        if label == 'person':
                            person_count += 1
                            # Draw Yellow Box for Person
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            except Exception as e:
                pass

        # 3. Weapon Detection (Separate pass with lower confidence)
        if self.model and self.show_suspicious:
            try:
                # Run inference for weapons with lower confidence
                results = self.model(frame, conf=WEAPON_CONFIDENCE, verbose=False)
                
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        label = self.model.names[cls]

                        # Detect weapons
                        if label in ['knife', 'scissors', 'baseball bat', 'gun', 'pistol']:
                            weapon_detected = True
                            weapon_name = label
                            # Draw Red Box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, f"WEAPON: {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            except Exception as e:
                pass

        # 4. Face Detection and Recognition
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3)
        
        # Draw Face Boxes and Recognize SRK (Only if Toggle is ON)
        if self.show_face and len(self.known_face_encodings) > 0:
            # Get face encodings from current frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare with known SRK encodings
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                color = (0, 255, 0)  # Green for unknown
                
                # If match found
                if True in matches:
                    name = "SRK FOUND!"
                    color = (0, 255, 255)  # Yellow for SRK
                    print(f"[ALERT] SRK DETECTED in frame!")
                    self.send_alert("ALERT: SRK has been detected!")
                
                # Draw box and label
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        elif self.show_face:
            # Fallback to simple face detection if no encodings loaded
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 5. Display Accurate Person Count
        if self.show_crowd:
             color = (0, 255, 255) # Yellow
             if person_count > 7: # Crowd limit > 7
                 color = (0, 0, 255) # Red
                 self.send_alert(f"SOS: OVERCROWD WARNING! {person_count} people detected (Limit: 7)")
             
             # Display Person Count at Top Left
             cv2.putText(frame, f"PEOPLE COUNT: {person_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
             print(f"[COUNT] People in frame: {person_count}")  # Console output

        # 6. Suspicious Alert (Weapon or Fight)
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
    print("   FINAL SECURITY SYSTEM (YOLOv8)  ")
    print("="*40)
    print(" [C] Toggle Crowd Counting")
    print(" [S] Toggle Suspicious Detection")
    print(" [F] Toggle Face Recognition")
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
