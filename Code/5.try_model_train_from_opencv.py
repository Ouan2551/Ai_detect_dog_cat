import cv2
import numpy as np
from tensorflow.keras.models import load_model #type: ignore

# load model from patch
loaded_model = load_model(r"D:\Important files Nannaphat\coding\Project\Open_cv_detect_dog_cat\model\dog_cat_model2.h5")

# output to check model
loaded_model.summary()

# class
characters = ["Dog", "Cat"]
image_size = (80, 80)
img = cv2.imread(r"D:\Important files Nannaphat\coding\Project\Open_cv_detect_dog_cat\dataset_train\val\images\7aa624b6-279.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, image_size)  # resize image
img = img.reshape(-1, image_size[0], image_size[1], 1)  # reshape image
img = img / 255.0  # normalize image

prediction = loaded_model.predict(img)
predicted_class = np.argmax(prediction)

print("Predicted class:", characters[predicted_class])