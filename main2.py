from ultralytics import YOLO
import cv2

# Load YOLOv8 model (COCO-trained or custom with cat/dog)
model = YOLO("yolov8n.pt")

# Load and resize image
image_path = "xa.jpg"  # Replace with your image file
image = cv2.imread(image_path)

# Resize to fixed frame size (640x480)
fixed_size = (640, 480)
image_resized = cv2.resize(image, fixed_size)

# Define allowed classes
allowed_classes = ['cat', 'dog']
class_names = model.names

# Run detection on resized image
results = model(image_resized)

# Loop through detections
for r in results:
    boxes = r.boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = class_names[cls_id]
            if class_name in allowed_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = f"{class_name} {conf:.2f}"

                # Draw rectangle and label
                cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image_resized, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Show result
cv2.imshow("YOLOv8 Detection - Cat & Dog", image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
