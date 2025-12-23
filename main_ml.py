import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os

class MLGestureVideoController:
    def __init__(self, model_path='models/gesture_classifier.pkl', 
                 scaler_path='models/scaler.pkl'):
        # Load trained model and scaler
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Model files not found!\n"
                f"Please run these steps first:\n"
                f"  1. python collect_data.py  (collect training data)\n"
                f"  2. python train_model.py   (train the model)"
            )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print("✓ Model and scaler loaded successfully")
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Video controller state
        self.video_state = 'paused'
        self.video_position = 0
        self.last_update = time.time()
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0
        
        # Current gesture tracking
        self.current_gesture = 'unknown'
        self.gesture_confidence = 0.0
        
        self.gesture_hold_start = None
        self.gesture_hold_duration = 3.0
        self.last_held_gesture = 'unknown'

        # Gesture display names
        self.gesture_display = {
            'play': 'Play ▶️',
            'pause': 'Pause ⏸️',
            'rewind': 'Rewind 30s ⏪',
            'fast_forward': 'Fast Forward 30s ⏩',
            'unknown': 'No Gesture'
        }

        self.cap = None
    
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmark coordinates as feature vector."""
        landmarks = []
        
        # Get wrist position for normalization
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # Extract all 21 landmarks
        for landmark in hand_landmarks.landmark:
            # Normalize coordinates relative to wrist
            landmarks.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])
        
        return np.array(landmarks).reshape(1, -1)
    
    def predict_gesture(self, hand_landmarks):
        """Predict gesture using trained ML model."""
        features = self.extract_landmarks(hand_landmarks)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction
        gesture = self.model.predict(features_scaled)[0]
        
        # Get confidence if model supports probability
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
        else:
            confidence = 1.0
        
        return gesture, confidence
    
    def update_video_position(self):
        """Update video position if playing."""
        if self.video_state == 'playing':
            current_time = time.time()
            elapsed = current_time - self.last_update
            self.video_position += elapsed
            self.last_update = current_time
    
    def execute_gesture(self, gesture, confidence):
        current_time = time.time()

        # Ignore unknown gestures
        if gesture == 'unknown':
            self.gesture_hold_start = None
            self.last_held_gesture = 'unknown'
            return False
        
        # Check if this is a new gesture or continuation of previous
        if gesture != self.last_held_gesture:
            # New gesture detected, start the hold timer
            self.gesture_hold_start = current_time
            self.last_held_gesture = gesture
            return False
        
        # Check if gesture has been held long enough
        if self.gesture_hold_start is None:
            self.gesture_hold_start = current_time
            return False
        
        hold_time = current_time - self.gesture_hold_start
        
        # If not held long enough, don't execute yet
        if hold_time < self.gesture_hold_duration:
            return False

        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return False
        
        action_executed = False
        
        if confidence > 0.9:
            if gesture == 'play':
                if self.video_state != 'playing':
                    self.video_state = 'playing'
                    self.last_update = time.time()
                    print(f"▶️  PLAYING (position: {self.format_time(self.video_position)})")
                    action_executed = True
            
            elif gesture == 'pause':
                if self.video_state != 'paused':
                    self.update_video_position()
                    self.video_state = 'paused'
                    print(f"⏸️  PAUSED (position: {self.format_time(self.video_position)})")
                    action_executed = True
            
            elif gesture == 'rewind':
                self.update_video_position()
                self.video_position = max(0, self.video_position - 30)
                print(f"⏪ REWIND 30s (position: {self.format_time(self.video_position)})")
                action_executed = True
            
            elif gesture == 'fast_forward':
                self.update_video_position()
                self.video_position += 30
                print(f"⏩ FAST FORWARD 30s (position: {self.format_time(self.video_position)})")
                action_executed = True
            
            if action_executed:
                self.last_gesture_time = current_time
        
        return action_executed
    
    def format_time(self, seconds):
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def draw_ui(self, frame):
        height, width = frame.shape[:2]
        
        # Update video position
        self.update_video_position()
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Video state
        state_text = f"State: {self.video_state.upper()}"
        state_color = (0, 255, 0) if self.video_state == 'playing' else (0, 165, 255)
        cv2.putText(frame, state_text, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        
        # Video position
        position_text = f"Position: {self.format_time(self.video_position)}"
        cv2.putText(frame, position_text, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Current gesture
        gesture_text = f"Gesture: {self.gesture_display.get(self.current_gesture, 'Unknown')}"
        gesture_color = (255, 100, 100) if self.current_gesture != 'unknown' else (128, 128, 128)
        cv2.putText(frame, gesture_text, (20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
        
        # Confidence
        confidence_text = f"Confidence: {self.gesture_confidence*100:.1f}%"
        cv2.putText(frame, confidence_text, (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Instructions at bottom
        instructions = [
            "ML-Powered Gesture Recognition | Press Q to quit"
        ]
        y_pos = height - 30
        cv2.putText(frame, instructions[0], (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def get_frame_for_web(self):
        """Get a single processed frame for web streaming."""
        success, frame = self.cap.read() if hasattr(self, 'cap') else (False, None)
        if not success:
            return None
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand
        result = self.hands.process(rgb_frame)
        
        # Detect and classify gesture
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Predict gesture
                gesture, confidence = self.predict_gesture(hand_landmarks)
                self.current_gesture = gesture
                self.gesture_confidence = confidence
                
                # Execute gesture
                self.execute_gesture(gesture, confidence)
        else:
            self.current_gesture = 'unknown'
            self.gesture_confidence = 0.0
        
        # Update video position
        self.update_video_position()
        
        # Encode frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_state(self):
        """Get current controller state for web UI."""
        return {
            'gesture': self.gesture_display.get(self.current_gesture, 'No Gesture'),
            'gesture_raw': self.current_gesture,
            'confidence': round(self.gesture_confidence * 100, 1),
            'video_state': self.video_state,
            'video_position': self.format_time(self.video_position),
            'status': self.video_state.capitalize()
        }

    def initialize_camera(self):
        """Initialize camera for web streaming."""
        if not hasattr(self, 'cap') or self.cap is None:
            self.cap = cv2.VideoCapture(0)
        return self.cap.isOpened()

    def run(self):
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*60)
        print("ML HAND GESTURE VIDEO CONTROLLER")
        print("="*60)
        print("\nGestures:")
        print("  Two fingers up     → Play")
        print("  Two fingers down   → Pause")
        print("  👍 Thumbs Up     → Fast Forward 30s")
        print("  👎 Thumbs Down   → Rewind 30s")
        print("\nPress 'q' to quit")
        print("="*60 + "\n")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand
            result = self.hands.process(rgb_frame)
            
            # Detect and classify gesture
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Predict gesture using ML model
                    gesture, confidence = self.predict_gesture(hand_landmarks)
                    self.current_gesture = gesture
                    self.gesture_confidence = confidence
                    
                    
                    self.execute_gesture(gesture, confidence)
            else:
                self.current_gesture = 'unknown'
                self.gesture_confidence = 0.0
            
            # Draw UI
            self.draw_ui(frame)
            
            # Display
            cv2.imshow("ML Gesture Video Controller", frame)
            
            # Check for quit
            if cv2.waitKey(1) == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\n✓ Video controller closed\n")

def main():
    try:
        controller = MLGestureVideoController()
        controller.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")

if __name__ == "__main__":
    main()
