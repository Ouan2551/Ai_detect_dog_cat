from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO(r"C:\Important files Nannaphat\coding\Project\runs\detect\train\weights\best.pt")

# Read image
img_path = r"C:\Important files Nannaphat\coding\Project\Machine_Learning\python+opencv\learn\image.jpg"
image = cv2.imread(img_path)

# Detect objects
results = model(image)

# Show results
for r in results:
    r.show()  # Display image with bounding boxes