import os
from imutils import paths
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

imagePaths = list(paths.list_images("Dataset\Faces"))
images = []
labels = []
labels_text = []
label_dict = {}

epochs = 50
activation_func = 'relu'
loss = "categorical_crossentropy"

label_num=0

for imagePath in imagePaths:

    img_class = imagePath.split("\\")[2]
    
    image = cv2.imread(imagePath)
    
    images.append(image)
    labels_text.append(img_class)

no_classes = len(np.unique(labels_text))

data = np.array(images, dtype="float") / 255.0

for label in labels_text:
    if label not in label_dict:
        label_dict[label] = label_num
        label_num+=1
        
    labels.append(label_dict[label])
    
labels = np.array(labels)

labels = tf.keras.utils.to_categorical(labels, num_classes=no_classes)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, labels, test_size=0.2, stratify=labels)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(8, kernel_size=(5,5), strides=(1,1), padding='same', activation=activation_func, input_shape=(160,160,3)))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(tf.keras.layers.Conv2D(16, kernel_size=(3,3), strides=(1,1), padding='same', activation=activation_func))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation=activation_func))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation=activation_func))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation=activation_func))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(2304, activation=activation_func))
model.add(tf.keras.layers.Dense(1152, activation=activation_func))
model.add(tf.keras.layers.Dense(576, activation=activation_func))
model.add(tf.keras.layers.Dense(144, activation=activation_func))

model.add(tf.keras.layers.Dense(no_classes, activation='softmax'))


model.compile(loss=loss, metrics=['accuracy'], optimizer='adam')

his = model.fit(Xtrain, Ytrain, batch_size=128, epochs=epochs, validation_data=(Xtest, Ytest))

model.save("model.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(range(epochs), his.history["loss"], label="train_loss")
plt.plot(range(epochs), his.history["val_loss"], label="val_loss")
plt.plot(range(epochs), his.history["accuracy"], label="train_acc")
plt.plot(range(epochs), his.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("model.png")