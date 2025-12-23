from flask import Flask, render_template, jsonify, request
import base64
import cv2
import numpy as np
from main_ml import MLGestureVideoController

app = Flask(__name__)

# Initialize gesture controller (no camera needed now)
try:
    gesture_controller = MLGestureVideoController()
    # Don't initialize camera - we'll process frames from client
    print("✓ Gesture controller initialized")
except Exception as e:
    print(f"❌ Failed to initialize gesture controller: {e}")
    gesture_controller = None

@app.route('/')
def index():
    """Landing page."""
    return render_template('index.html')

@app.route('/control')
def control():
    """Main control page."""
    if gesture_controller is None:
        return "Error: Gesture controller not initialized. Please check model files.", 500
    return render_template('control.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process frame from client-side camera."""
    if gesture_controller is None:
        return jsonify({
            'gesture': 'Controller not initialized',
            'confidence': 0,
            'video_state': 'error',
            'video_position': '00:00'
        }), 500
    
    try:
        # Get image data from request
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove "data:image/jpeg;base64,"
        
        # Decode base64 to image
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Process frame with gesture controller
        # You'll need to modify your MLGestureVideoController to have a method
        # that processes a single frame instead of reading from camera
        state = gesture_controller.process_frame(frame)
        
        return jsonify(state)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset_controller')
def reset_controller():
    """Reset the gesture controller state."""
    if gesture_controller:
        gesture_controller.video_state = 'paused'
        gesture_controller.video_position = 0
        gesture_controller.current_gesture = 'unknown'
        gesture_controller.gesture_confidence = 0.0
        gesture_controller.gesture_hold_start = None
        gesture_controller.last_held_gesture = 'unknown'
        return jsonify({'status': 'reset'})
    return jsonify({'status': 'error'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("GESTURE MUSIC CONTROL WEB APP")
    print("="*60)
    print("\nStarting server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("\nGestures:")
    print("  Two fingers up     → Play")
    print("  Two fingers down   → Pause")
    print("  👍 Thumbs Up       → Fast Forward 30s")
    print("  👎 Thumbs Down     → Rewind 30s")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    try:
        app.run(debug=True, threaded=True, use_reloader=False, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\n✓ Server stopped")