from ultralytics import YOLO
import cv2

# Load custom YOLOv8 model
model = YOLO("best.pt")

# Manually define class names (based on your training labels)
class_names = {0: 'cat', 1: 'dog'}
allowed_class_ids = [0, 1]  # Only allow detection of these classes

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in allowed_class_ids:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = f"{class_names[cls_id]} {conf:.2f}"

                    # Draw rectangle and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("YOLOv8 Detection - Cat & Dog", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
