import os
from imutils import paths
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2

imagePaths = list(paths.list_images("Dataset\Faces"))
images = []
classes = []

for imagePath in imagePaths:

    img_class = imagePath.split("\\")[2]
    
    image = cv2.imread(imagePath)
    
    images.append(image)
    classes.append(img_class)

np.array(images, dtype="float") / 255.0

no_classes = len(np.unique(classes))

Xtrain, Xtest, Ytrain, Ytest = train_test_split(images, classes, test_size=0.2, stratify=classes)

model = tf.keras.models.Sequential()

#TODO
model.add(tf.keras.layers.Conv2D())

print(no_classes)

# print("Xtrain shape:", np.shape(Xtrain))
# print("Xtest shape:", np.shape(Xtest))
# print("Ytrain shape:", Ytrain.shape)
# print("Ytest shape:", Ytest.shape)

# img = cv2.imread("Dataset\Faces\Alexandra Daddario\Alexandra Daddario_1.jpg")

# cv2.imshow("image", img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()