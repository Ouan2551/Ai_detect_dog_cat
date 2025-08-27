import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model #type: ignore

image_size = (80, 80)
channels = 1

loaded_model = load_model(r"D:\Important files Nannaphat\coding\Project\Ai_detect_dog_cat\model\dog_cat_model_tf.h5")
# loaded_model.summary()
categories = ['Dog', 'Cat']

def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, image_size)
    img = img.reshape(1, 80, 80, 1) / 255.0
    pred = loaded_model.predict(img)[0]
    return categories[np.argmax(pred)]

# directory of picture for testing model
# print(predict_image(r'D:\Important files Nannaphat\coding\Project\Ai_detect_dog_cat\Data\Dog\9.jpg'))
# print(predict_image(r"D:\Important files Nannaphat\coding\Project\Ai_detect_dog_cat\dataset_train\val\images\4a37ca86-61.jpg"))
print(predict_image(r"D:\quick_share\IMG_20250308_141318.jpg"))