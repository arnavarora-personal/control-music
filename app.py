from flask import Flask, render_template, Response, jsonify
import threading
from main_ml import MLGestureVideoController

app = Flask(__name__)

# Initialize gesture controller
try:
    gesture_controller = MLGestureVideoController()
    gesture_controller.initialize_camera()
    print("✓ Gesture controller initialized")
except Exception as e:
    print(f"❌ Failed to initialize gesture controller: {e}")
    gesture_controller = None

# Thread lock for camera access
camera_lock = threading.Lock()

def gen_frames():
    """Generate frames for video streaming."""
    if gesture_controller is None:
        return
    
    while True:
        with camera_lock:
            frame = gesture_controller.get_frame_for_web()
        
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_gesture')
def get_gesture():
    """Get current gesture state as JSON."""
    if gesture_controller is None:
        return jsonify({
            'gesture': 'Controller not initialized',
            'confidence': 0,
            'video_state': 'error',
            'video_position': '00:00',
            'status': 'Error'
        })
    
    with camera_lock:
        state = gesture_controller.get_state()
    
    return jsonify(state)

@app.route('/reset_controller')
def reset_controller():
    """Reset the gesture controller state."""
    if gesture_controller:
        with camera_lock:
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
        app.run(debug=True, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n✓ Server stopped")
    finally:
        if gesture_controller and hasattr(gesture_controller, 'cap'):
            gesture_controller.cap.release()