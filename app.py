import os
from imutils import paths
from sklearn.model_selection import train_test_split
import cv2

imagePaths = list(paths.list_images("Dataset\Faces"))
classes = []

for image in imagePaths:

    img_class = image.split("\\")[2]
    
    classes.append(img_class)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(imagePaths, classes, test_size=0.2, stratify=classes)

    # img = cv2.imread("Dataset\Faces\Alexandra Daddario\Alexandra Daddario_1.jpg")

    # cv2.imshow("image", img)

    # cv2.waitKey(0)

    # cv2.destroyAllWindows()