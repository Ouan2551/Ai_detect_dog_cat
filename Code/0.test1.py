import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

# โหลดโมเดลที่เทรนไว้
model = load_model(r"C:\Important files Nannaphat\coding\Project\Open_cv_detect_dog_cat\model\dog_cat_model.h5")  # แก้ path

classes = ["Not Person", "Person"]
image_size = (80, 80)

# ใช้ HOG + SVM Detector จาก OpenCV เพื่อตรวจจับ "รูปร่างคน"
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_frame = frame.copy()

    # ตรวจจับตำแหน่งคนในภาพ
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)

    for (x, y, w, h) in boxes:
        # Crop เฉพาะส่วนที่ตรวจเจอ
        person_roi = frame[y:y+h, x:x+w]

        # แปลงเป็น grayscale และ resize ตามที่โมเดลต้องการ
        try:
            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, image_size)
            normalized = resized / 255.0
            reshaped = normalized.reshape(1, image_size[0], image_size[1], 1)

            # ทำนายว่าเป็นคนจริงหรือไม่
            prediction = model.predict(reshaped, verbose=0)
            predicted_class = np.argmax(prediction)
            label = classes[predicted_class]

            # วาดกรอบ
            color = (0, 255, 0) if label == "Person" else (0, 0, 255)
            cv2.rectangle(orig_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(orig_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        except Exception as e:
            print(f"Error processing ROI: {e}")
            continue

    # แสดงผล
    cv2.imshow("Real-time Person Detection", orig_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
