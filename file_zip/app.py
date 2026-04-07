import cv2
import os
import numpy as np
import base64
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory
from ultralytics import YOLO

app = Flask(__name__, template_folder='.')

# --- STORAGE ---
DETECTION_FOLDER = 'static/detections'
if not os.path.exists(DETECTION_FOLDER):
    os.makedirs(DETECTION_FOLDER)

# Load Models
pothole_model = YOLO('custom.pt')
object_model = YOLO('yolov8n.pt')

detection_log = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_live_frame', methods=['POST'])
def process_live_frame():
    global detection_log
    start_time = time.time() # Start timing inference
    
    try:
        data = request.get_json()
        image_data = data['image'].split(",")[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        alert_triggered = False
        message = "Road Clear"

        # AI Detection
        p_results = pothole_model(frame, conf=0.5, verbose=False)
        if len(p_results[0].boxes) > 0:
            alert_triggered = True
            message = "POTHOLE AHEAD"
        else:
            o_results = object_model(frame, conf=0.5, verbose=False)
            if len(o_results[0].boxes) > 0:
                label = object_model.names[int(o_results[0].boxes[0].cls[0])]
                if label in ['person', 'car', 'truck', 'bus']:
                    alert_triggered = True
                    message = f"{label.upper()} DETECTED"

        # Calculate how long the AI took (in milliseconds)
        inference_ms = (time.time() - start_time) * 1000

        if alert_triggered:
            filename = f"det_{int(time.time())}.jpg"
            cv2.imwrite(os.path.join(DETECTION_FOLDER, filename), frame)
            log_entry = {
                "time": datetime.now().strftime("%H:%M:%S"), 
                "msg": message, 
                "url": f"/static/detections/{filename}"
            }
            if not detection_log or detection_log[0]['msg'] != message:
                detection_log.insert(0, log_entry)

        return jsonify({
            "alert": alert_triggered, 
            "message": message,
            "inf_ms": round(inference_ms, 1)
        })
    except:
        return jsonify({"alert": False, "inf_ms": 0})

@app.route('/get_history')
def get_history():
    return jsonify(detection_log)

@app.route('/delete_history', methods=['POST'])
def delete_history():
    global detection_log
    detection_log = []
    for f in os.listdir(DETECTION_FOLDER):
        os.remove(os.path.join(DETECTION_FOLDER, f))
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)