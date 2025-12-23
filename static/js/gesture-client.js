// Get elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const detectedGestureEl = document.getElementById('detected-gesture');
const confidenceEl = document.getElementById('confidence-percent');
const videoPositionEl = document.getElementById('video-position');
const playbackStatusEl = document.getElementById('playback-status');
const statusIndicator = document.getElementById('status');

let isProcessing = false;

// Start webcam
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: 640, 
                height: 480 
            } 
        });
        video.srcObject = stream;
        statusIndicator.querySelector('span:last-child').textContent = 'Camera Active';
        statusIndicator.querySelector('.status-dot').style.backgroundColor = '#10b981';
        
        // Start sending frames after video is ready
        video.addEventListener('loadeddata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            sendFrames();
        });
    } catch (error) {
        console.error('Error accessing webcam:', error);
        statusIndicator.querySelector('span:last-child').textContent = 'Camera Error';
        statusIndicator.querySelector('.status-dot').style.backgroundColor = '#ef4444';
        alert('Could not access camera. Please ensure you have granted camera permissions.');
    }
}

// Capture and send frames to server
function sendFrames() {
    setInterval(async () => {
        if (isProcessing) return; // Skip if still processing previous frame
        
        isProcessing = true;
        
        // Draw current video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64 JPEG
        const imageData = canvas.toDataURL('image/jpeg', 0.7);
        
        try {
            // Send to Flask server
            const response = await fetch('/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            
            if (response.ok) {
                const data = await response.json();
                updateUI(data);
            }
        } catch (error) {
            console.error('Error sending frame:', error);
        } finally {
            isProcessing = false;
        }
    }, 100); // Send ~10 frames per second
}

// Update UI with gesture data
function updateUI(data) {
    // Update gesture display
    detectedGestureEl.textContent = data.gesture || 'No gesture detected';
    
    // Update confidence
    const confidence = Math.round((data.confidence || 0) * 100);
    confidenceEl.textContent = `${confidence}%`;
    
    // Update video position
    videoPositionEl.textContent = data.video_position || '00:00';
    
    // Update playback status
    const status = data.video_state || 'paused';
    playbackStatusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    playbackStatusEl.style.color = status === 'playing' ? '#10b981' : '#6b7280';
}

// Start when page loads
startWebcam();