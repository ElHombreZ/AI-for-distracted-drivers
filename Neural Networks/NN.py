import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers

### understanding models
#creating 1 layer model
model_1 = Sequential() #info flows from left to right
model_1.add(Dense(4, input_shape=(3,),activation = 'relu')) #Dense = every neuron received input from previous neuron and outputs to next neuron. 'relu' outputs only positive numbers. layer of 4 neurons. 3 neuron inputs
model_1.add(Dense(1, activation = 'linear')) #'linear' outputs any number
model_1.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['mean_squared_error'])

#creating 2 layer models
model_2 = Sequential()
model_2.add(Dense(4, input_shape = (3,), activation = 'relu'))
model_2.add(Dense(2, activation = 'softmax')) #'softmax outputs 0 to 1

model_2.compile(loss='categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

#building first model for images
model_3 = Sequential()
model_3.add(Flatten(input_shape = (64, 64, 3)))
model_3.add(Dense(units = 128, activation = 'relu'))
model_3.add(Dense(units = 4, activation = 'softmax'))
model_3.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.95),
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint

monitor = ModelCheckpoint('./model.h5', monitor='val_loss', verbose=0,
                          save_best_only=True, save_weights_only=False,
                          mode='auto', save_freq='epoch')

data_augmentation = ImageDataGenerator(
    rotation_range=10,  # Rotate images randomly up to 10 degrees
    width_shift_range=0.1,  # Shift images horizontally by a fraction of the width
    height_shift_range=0.1,  # Shift images vertically by a fraction of the height
    shear_range=0.2,  # Apply shear transformation with a shear angle up to 20 degrees
    zoom_range=0.2,  # Randomly zoom images by a factor up to 20%
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest'  # Fill in newly created pixels after rotation or shifting
)

X_train, y_train = get_train_data()
X_test, y_test = get_test_data()

X_train = X_train.reshape([-1, 64, 64, 3])
X_test = X_test.reshape([-1, 64, 64, 3])

y_train = label_to_numpy(y_train)
y_test = label_to_numpy(y_test)

#fitting model
history = model_3.fit(data_augmentation.flow(X_train, y_train, batch_size=32), epochs = 10, validation_data = (X_test, y_test), shuffle = True, callbacks = [monitor])

#val acc is the acccuracy in testing, acc is the accuracy in training. Model is underfitting

#plotting model
plot_acc(history)

#Compiling and training with a convolutional neural network
cnn = Sequential()
cnn.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3)))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dense(units = 4, activation = 'softmax'))

# compile the network
cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.SGD(learning_rate=1e-3, momentum=0.95),
            metrics=['accuracy'])

#train
ata_augmentation = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
history = cnn.fit(data_augmentation.flow(X_train, y_train, batch_size=32), epochs = 50, validation_data = (X_test, y_test), shuffle = True, callbacks = [monitor])

#val acc is the acccuracy in testing, acc is the accuracy in training. Model is overfitting (in 50 epochs)

plot_acc(history)

### with built in models such as VGG16
(X_train, y_train) = get_train_data(flatten=True)
(X_test, y_test) = get_test_data(flatten=True)

X_train = X_train.reshape([-1, 64, 64, 3])
X_test = X_test.reshape([-1, 64, 64, 3])

y_train = label_to_numpy(y_train)
y_test = label_to_numpy(y_test)

transfer = TransferClassifier(name = 'VGG16') #or VGG19, ResNet50, DenseNet121


history = transfer.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test), shuffle = True, callbacks = [monitor])
plot_acc(history) #still overfitting


from tensorflow.keras.applications.vgg16 import VGG16
vgg_expert = VGG16(weights = 'imagenet', include_top = False, input_shape = (64, 64, 3))
vgg_model = Sequential()
vgg_model.add(vgg_expert)

#adding more layers
vgg_model = Sequential()
vgg_model.add(vgg_expert)
vgg_model.add(GlobalAveragePooling2D())
vgg_model.add(Dense(1024, activation = 'relu'))
vgg_model.add(Dropout(0.3))
vgg_model.add(Dense(512, activation = 'relu'))
vgg_model.add(Dropout(0.3))
vgg_model.add(Dense(4, activation = 'softmax'))

#compiling model
vgg_model.compile(loss='categorical_crossentropy',
            optimizer=optimizers.SGD(learning_rate = 1e-4, momentum = 0.95),
            metrics='accuracy')

#training again
(X_train, y_train) = get_train_data(flatten=True)
(X_test, y_test) = get_test_data(flatten=True)

X_train = X_train.reshape([-1, 64, 64, 3])
X_test = X_test.reshape([-1, 64, 64, 3])

y_train = label_to_numpy(y_train)
y_test = label_to_numpy(y_test)

history = vgg_model.fit(X_train, y_train, epochs = 30, validation_data = (X_test, y_test), shuffle = True, callbacks = [monitor])
#still overfitting but much better

plot_acc(history)


