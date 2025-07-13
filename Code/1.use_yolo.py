from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")  # 'yolov8n.pt' is a pre-trained model

# Load image
image = cv2.imread(r"D:\Important files Nannaphat\coding\Project\Open_cv_detect_dog_cat\Data_Set\Dog\2.jpg")

# Run YOLO detection
results = model(image)

# Draw bounding boxes
for result in results:
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show the output
cv2.imshow("YOLO Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()