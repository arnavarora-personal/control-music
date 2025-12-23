// Update gesture display in real-time
window.addEventListener('DOMContentLoaded', () => {
    // Reset backend state
    fetch('/reset_controller')
        .then(response => response.json())
        .then(data => {
            console.log('Controller reset:', data.status);
            resetUI();
        })
        .catch(error => {
            console.error('Error resetting controller:', error);
            resetUI(); // Still reset UI even if backend fails
        });
});

function resetUI() {
    // Reset gesture display
    document.getElementById('detected-gesture').textContent = 'No gesture detected';
    
    // Reset video position
    const positionElement = document.getElementById('video-position');
    if (positionElement) {
        positionElement.textContent = '00:00';
    }
    
    // Reset confidence
    const confidenceElement = document.getElementById('confidence-percent');
    if (confidenceElement) {
        confidenceElement.textContent = '0%';
    }
    
    // Reset playback status
    const statusElement = document.getElementById('playback-status');
    if (statusElement) {
        statusElement.textContent = 'Paused';
        statusElement.style.color = '#f59e0b';
    }
    
    // Reset status indicator
    const statusDot = document.querySelector('.status-dot');
    if (statusDot) {
        statusDot.style.background = '#f59e0b';
    }
    
    console.log('UI reset complete');
}


const updateGesture = () => {
    fetch('/get_gesture')
        .then(response => response.json())
        .then(data => {
            // Update gesture display
            document.getElementById('detected-gesture').textContent = data.gesture || 'No gesture detected';
            
            // Update confidence
            const confidenceElement = document.getElementById('confidence-percent');
            if (confidenceElement) {
                confidenceElement.textContent = data.confidence + '%';
            }
            
            // Update video position
            const positionElement = document.getElementById('video-position');
            if (positionElement) {
                positionElement.textContent = data.video_position || '00:00';
            }
            
            // Update playback status
            const statusElement = document.getElementById('playback-status');
            if (statusElement) {
                statusElement.textContent = data.status || 'Unknown';
                
                // Color code the status
                if (data.status === 'Playing') {
                    statusElement.style.color = '#10b981';
                } else if (data.status === 'Paused') {
                    statusElement.style.color = '#f59e0b';
                } else {
                    statusElement.style.color = '#ef4444';
                }
            }
            
            // Update status indicator
            const statusDot = document.querySelector('.status-dot');
            if (statusDot && data.video_state === 'playing') {
                statusDot.style.background = '#10b981';
            } else if (statusDot) {
                statusDot.style.background = '#f59e0b';
            }
        })
        .catch(error => console.error('Error fetching gesture data:', error));
};

// Poll for updates every 300ms
setInterval(updateGesture, 300);

setTimeout(() => {
    updateGesture();
}, 1000);