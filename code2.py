from ultralytics import YOLO
import cv2

# Load custom YOLOv8 model
model = YOLO("best.pt")

# Define class names manually
class_names = {0: 'cat', 1: 'dog'}
allowed_class_ids = [0, 1]

# Load and resize the image to a fixed size
image_path = "zcz.jpg"  # Replace with your image file path
img = cv2.imread(image_path)
resized_img = cv2.resize(img, (640, 640))  # Resize to 640x640

# Run detection
results = model(resized_img)

# Process results
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
                cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(resized_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Show result
cv2.imshow("YOLOv8 Image Detection - Cat & Dog", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
