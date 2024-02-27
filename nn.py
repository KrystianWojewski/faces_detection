from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, AveragePooling2D, Dropout

class MyModel:
    
    def build(size, num_classes):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu', input_shape=(size, size, 3)))
        model.add(AveragePooling2D(pool_size=(2,2), strides=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))
        model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        model.add(AveragePooling2D(pool_size=(2,2), strides=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))
        model.add(Conv2D(128, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))
        model.add(AveragePooling2D(pool_size=(2,2), strides=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        model.add(AveragePooling2D(pool_size=(2,2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(250, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(num_classes, activation='softmax'))

        return model
    
class VGG16:
    def build(size, num_classes):
        model = Sequential()
        model.add(Conv2D(input_shape=(size,size,3),filters=32,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=num_classes, activation="softmax"))
        
        return model