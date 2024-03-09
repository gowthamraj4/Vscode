from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

# 2D convolutional layer as the first layer in the model,
# equipped with 32 filters /kernels of size 3x3 each,
# uses the ReLU activation function, and expects input images of shape  = (64,64, 3)


model.add(Conv2D(32,(3,3), input_shape = (64,64, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(units= 128,activation='relu'))
model.add(Dense(units=1,activation='sigmoid')) # ! neuron , output will be either true or false

model.compile(optimizer='adam',loss='binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale= 1./255 , shear_range= 0.2 , zoom_range= 0.2 , horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale= 1./255)

training_set = train_datagen.flow_from_dataframe('Dataset/train'
                                                 target_size=(64,64)
                                                 batch_size= 8
                                                 class_mode= 'binary')

val_set = test_datagen.flow_from_dataframe('Dataset/val'
                                                 target_size=(64,64)
                                                 batch_size= 8
                                                 class_mode= 'binary')

model.fit_generator()