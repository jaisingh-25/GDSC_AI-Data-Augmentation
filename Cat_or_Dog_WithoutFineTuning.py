import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_augmentation=ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator=train_augmentation.flow_from_directory('.../dataset/Train', target_size=(128,128), batch_size=20, class_mode='binary')
validation_augmentation=ImageDataGenerator(rescale=1./255)
validation_generator=validation_augmentation.flow_from_directory('.../dataset/Test', target_size=(128,128), batch_size=20, class_mode='binary')
conv_base=VGG16(input_shape=(128,128,3), include_top=False, weights='imagenet')

for layer in conv_base.layers:
    layer.trainable=False
    
model = Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(units= 256, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit(train_generator, steps_per_epoch=10, epochs=10, verbose=1,
          validation_data=validation_generator)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
