import cv2
import numpy as np
from ultralytics import YOLO
import torch

# ========== Lane Detection ========== #
def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(0, height), (img.shape[1], height), (img.shape[1]//2, int(height*0.6))]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped_edges = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped_edges,
        2,
        np.pi / 180,
        50,
        np.array([]),
        minLineLength=40,
        maxLineGap=150
    )

    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

# ========== Main Pipeline ========== #
def main():
    model = YOLO("yolov8n.pt")  # pretrained COCO model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    video_path = r"E:\creva\video testing demo\5194433-hd_1920_1080_30fps (1).mp4"  # change if needed
    cap = cv2.VideoCapture(video_path)

    target_width, target_height = 640, 360  # resize for speed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (target_width, target_height))

        # Detect lanes
        lane_frame = detect_lanes(frame)

        # YOLO object detection
        results = model(lane_frame, stream=True, verbose=False)
        for r in results:
            annotated = r.plot()  # draw boxes on frame

        # Show final combined output
        cv2.imshow("YOLOv8 + Lane Detection", annotated)

        # Quit if 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
