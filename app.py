import os
from imutils import paths
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import nn

# Disable TF warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

imagePaths = list(paths.list_images("Dataset\Faces"))
images = []
labels = []
labels_text = []
label_dict = {}

epochs = 50
loss = "categorical_crossentropy"

size = 160

doResize = False
re_size = 80

label_num=0

for imagePath in imagePaths:

    img_class = imagePath.split("\\")[2]
    
    image = cv2.imread(imagePath)
    
    if doResize:
        size = re_size
        image = cv2.resize(image, (size , size))
    
    images.append(image)
    labels_text.append(img_class)

num_classes = len(np.unique(labels_text))

data = np.array(images, dtype="float") / 255.0

for label in labels_text:
    if label not in label_dict:
        label_dict[label] = label_num
        label_num+=1
        
    labels.append(label_dict[label])
    
labels = np.array(labels)

labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, labels, test_size=0.2, stratify=labels)

model = nn.MyModel.build(size, num_classes)

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
plt.savefig("model7.png")