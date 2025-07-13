from ultralytics import YOLO

# Load YOLO model (pretrained)
model = YOLO("yolov8n.pt")

# Train the model
model.train(data=r"D:\Important files Nannaphat\coding\Project\Open_cv_detect_dog_cat\dataset_train\data.yml", epochs=50, imgsz=640, batch=16)

# Save the trained model
model.export(format="onnx")  # Export to ONNX for deployment (optional)