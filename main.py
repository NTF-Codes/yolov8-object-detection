from ultralytics import YOLO
import cv2

# Load YOLOv8 model (replace 'best.pt' with 'yolov8n.pt', 'yolov8s.pt', etc. if needed)
model = YOLO("yolov8n.pt")  # Ensure this model includes 'cat' and 'dog' classes (COCO trained)

# Define class names to detect
allowed_classes = ['cat', 'dog']
class_names = model.names  # class_names is a dict with index: name

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
                class_name = class_names[cls_id]
                if class_name in allowed_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = float(box.conf[0])
                    label = f"{class_name} {conf:.2f}"

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
