import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

def prepare_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        # Load the image from disk
        img = cv2.imread(os.path.join(image_dir, filename))
        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize the image to a fixed size
        img_resized = cv2.resize(img, (128, 128))
        # Convert the resized image to a NumPy array
        img_array = np.array(img_resized)
        # Normalize the pixel values to be between 0 and 1
        img_normalized = img_array / 255.0
        # Append the normalized image to the list
        images.append(img_normalized)
    # Convert the list of images to a NumPy array
    images = np.array(images)
    # Reshape the array to have a single channel
    images = np.reshape(images, (*images.shape, 1))
    return images

def load_images(image_dir):
    images = []
    count = 0
    for filename in os.listdir(image_dir):
        # Load the image from disk
        img = cv2.imread(os.path.join(image_dir, filename))
        # Convert the image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Append the image to the list
        images.append(img)
        
        count = count + 1
        if count%1000 == 0:
            print(f"Loading images, currently on {count}.")
        
    # Convert the list of images to a NumPy array
    images = np.array(images)
    return images

def preprocess_images(images):
    # Resize the images to 128x128
    images = [cv2.resize(img, (128, 128)) for img in images]
    
    # Convert the images to grayscale
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    
    # Reshape the images to (128, 128, 1) format
    images = np.array(images)
    images = np.reshape(images, (-1, 128, 128, 1))
    
    # Convert the pixel values to floats between 0 and 1
    images = images.astype('float32') / 255
    
    return images
  
  # Load the images
images = load_images("../Dataset/img_align_celeba/img_align_celeba")

print("Images Loaded")

np.save('images_array.npy', images)

#images = np.load("../Dataset/images_array.npy", allow_pickle=True)

# Load the labels
df_labels = pd.read_csv("../Dataset/list_attr_celeba.csv")
XY_labels = df_labels['Male'].values
XY_labels = (XY_labels + 1)/2

XY_labels_train = XY_labels[:150000]
XY_labels_test = XY_labels[150000:]

XY_images_train = images[:150000]
XY_images_test = images[150000:]

XY_images_train = preprocess_images(XY_images_train)
XY_images_test = preprocess_images(XY_images_test)

print("Starting Model")

# Define the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Training the Model")

# Train the model
model.fit(XY_images_train, XY_labels_train, epochs=1, validation_data=(XY_images_test, XY_labels_test))

print("Training Done")

model.save("my_model.h5")

# Evaluate the model
test_loss, test_acc = model.evaluate(XY_images_test, XY_labels_test)
print('Test accuracy:', test_acc)
