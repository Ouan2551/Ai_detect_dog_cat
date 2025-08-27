import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical #type: ignore

# setup picture
image_size = (80, 80)
channels = 1

# path of picture
dataset_path = r"D:\Important files Nannaphat\coding\Project\Ai_detect_dog_cat\Data"
categories = ['Dog', 'Cat']

data = []
labels = []

for idx, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale
        if img is not None:
            img = cv2.resize(img, image_size)
            data.append(img)
            labels.append(idx)

data = np.array(data).reshape(-1, image_size[0], image_size[1], 1) / 255.0
labels = to_categorical(labels, num_classes=2)

# train/test
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
# random_state using for random weight, algorithm, batch (same every time)

# create cnn model
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout #type: ignore

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # output 2 class
])

from tensorflow.keras.optimizers import Adam #type: ignore
# learning_rate = 0.01; optimizer = Adam(learning_rate=learning_rate) # custom learning rate (constant value)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# adam using constant value (learning_rate = 0.001)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# train model
EPOCHS = 30
BATCH_SIZE = 32

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# save model
model.save(r"D:\Important files Nannaphat\coding\Project\Ai_detect_dog_cat\model\dog_cat_model_tf1.h5")

# graph accuracy and loss
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Training vs Validation Accuracy")
# plt.legend()
# plt.grid(True)
# plt.show()