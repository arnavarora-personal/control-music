import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

class GestureDataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.gestures = ['play', 'pause', 'rewind', 'fast_forward']
        self.current_gesture_idx = 0
        
        self.data = []
        self.labels = []
        
        self.samples_per_gesture = 100
        self.samples_collected = 0
        
    def extract_landmarks(self, hand_landmarks):
        landmarks = []
        
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])
        
        return np.array(landmarks)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*60)
        print("HAND GESTURE DATA COLLECTOR")
        print("="*60)
        print("\nGestures to collect:")
        print("  1. PLAY: Open palm (5 fingers up)")
        print("  2. PAUSE: Closed fist (0 fingers)")
        print("  3. REWIND: Thumbs down")
        print("  4. FAST FORWARD: Thumbs up")
        print("\nInstructions:")
        print("  - Press SPACE to capture current gesture")
        print("  - Press 'n' to move to next gesture")
        print("  - Press 's' to save data")
        print("  - Press 'q' to quit")
        print("="*60 + "\n")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand
            result = self.hands.process(rgb_frame)
            
            # Current gesture info
            current_gesture = self.gestures[self.current_gesture_idx]
            progress = f"{self.samples_collected}/{self.samples_per_gesture}"
            
            # Draw UI
            height, width = frame.shape[:2]
            
            # Info panel
            cv2.rectangle(frame, (10, 10), (width - 10, 160), (0, 0, 0), -1)
            cv2.putText(frame, f"Collecting: {current_gesture.upper()}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(frame, f"Progress: {progress}", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Total samples: {len(self.data)}", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Instructions
            instructions = [
                "SPACE=Capture | N=Next gesture | S=Save | Q=Quit"
            ]
            y_pos = height - 30
            cv2.putText(frame, instructions[0], (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw hand landmarks
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Show that hand is detected
                    cv2.putText(frame, "HAND DETECTED", (width - 250, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "NO HAND DETECTED", (width - 280, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Gesture Data Collector", frame)
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - capture sample
                if result.multi_hand_landmarks:
                    landmarks = self.extract_landmarks(result.multi_hand_landmarks[0])
                    self.data.append(landmarks)
                    self.labels.append(current_gesture)
                    self.samples_collected += 1
                    print(f"✓ Captured {current_gesture} sample {self.samples_collected}")
                    
                    # Auto-advance after collecting enough samples
                    if self.samples_collected >= self.samples_per_gesture:
                        self.samples_collected = 0
                        self.current_gesture_idx = (self.current_gesture_idx + 1) % len(self.gestures)
                        if self.current_gesture_idx == 0:
                            print("\n✓ Collected all gestures! Press 's' to save.")
                else:
                    print("✗ No hand detected!")
            
            elif key == ord('n'):  # Next gesture
                self.samples_collected = 0
                self.current_gesture_idx = (self.current_gesture_idx + 1) % len(self.gestures)
                print(f"\nSwitched to: {self.gestures[self.current_gesture_idx]}")
            
            elif key == ord('s'):  # Save data
                self.save_data()
            
            elif key == ord('q'):  # Quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
    
    def save_data(self):
        """Save collected data to file."""
        if len(self.data) == 0:
            print("No data to save!")
            return
        
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/gesture_data_{timestamp}.npz"
        
        # Save as numpy arrays
        np.savez(
            filename,
            data=np.array(self.data),
            labels=np.array(self.labels)
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Data saved to: {filename}")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Features per sample: {len(self.data[0])}")
        
        # Show distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"\nSample distribution:")
        for gesture, count in zip(unique, counts):
            print(f"  {gesture}: {count} samples")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    collector = GestureDataCollector()
    collector.run()
