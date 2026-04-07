import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone
import os
import datetime


def detect(source=0):
    """
    Multi-object detection + pothole detection using YOLO.
    Args:
        source: webcam index (0) or path to a video file.
    """

    base_path = os.path.dirname(__file__)

    # Load models
    pothole_model = YOLO(os.path.join(base_path, 'custom.pt'))          # your pothole model
    object_model = YOLO('yolov8n.pt')                                   # general object detection

    # Open video stream
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Error: cannot open source: {source}")
        return

    tracker = Tracker()

    # YOLO built-in COCO class names
    coco_classes = object_model.names
    pothole_classes = ["pothole"]

    current_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    data = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🔚 End of video stream.")
            break

        # Run both detections
        pothole_results = pothole_model(frame, verbose=False)
        object_results = object_model(frame, verbose=False)

        # -------------------------
        # 🔹 Draw Pothole Detections
        # -------------------------
        for r in pothole_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = pothole_classes[cls]

                # Blue for potholes
                cvzone.cornerRect(frame, (x1, y1, w, h), l=8, colorC=(255, 0, 0))
                cvzone.putTextRect(
                    frame,
                    f"{cls_name} {conf:.2f}",
                    (x1, max(30, y1)),
                    scale=1,
                    thickness=1,
                    colorT=(255, 0, 0)
                )

        # ----------------------------------
        # 🔹 Draw General Object Detections
        # ----------------------------------
        for r in object_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = coco_classes[cls]

                # Skip pothole class here (handled above)
                if cls_name.lower() == "pothole":
                    continue

                # Green for objects (cars, people, bikes, trucks...)
                cvzone.cornerRect(frame, (x1, y1, w, h), l=8, colorC=(0, 255, 0))
                cvzone.putTextRect(
                    frame,
                    f"{cls_name} {conf:.2f}",
                    (x1, max(30, y1)),
                    scale=1,
                    thickness=1,
                    colorT=(0, 255, 0)
                )

        # -------------------------
        # Display
        # -------------------------
        cv2.imshow("Pothole + Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save CSV if needed
    if data:
        df = pd.DataFrame.from_dict(data, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)
        df.to_csv(f"{current_date}.csv", index=False)

    cap.release()
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    # Auto select video or webcam
    video_path = r"E:\creva\video testing demo\5194433-hd_1920_1080_30fps (1).mp4"
    source = video_path if os.path.exists(video_path) else 0
    detect(source)
