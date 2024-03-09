from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization, Dropout


model = Sequential()
model.add(Conv2D(32 , kernel_size=(3,3) , activation = 'relu' , input_shape = ( 128 , 128 , 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64 , kernel_size=(3,3) , activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64 , kernel_size=(3,3) , activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(96 , kernel_size=(3,3) , activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32 , kernel_size=(3,3) , activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

# to help prevent overfitting by randomly setting a fraction of input units to zero during training.
model.add(Dropout(0.2))

# The Flatten layer is used to flatten the 2D arrays to a vector before the fully connected Dense layer with 128 units.
model.add(Flatten())
model.add(Dense(128 , activation = 'relu'))

# ReLU is often used in hidden layers of a neural network for introducing non-linearity
# softmax is used in the output layer for multi-class classification to obtain probability distributions.
model.add(Dropout(0.3))
model.add(Dense(25, activation='softmax'))

model.compile(optimizer='adam',loss= 'categorical_crossentropy', metrics= ['accuracy'] )

# Data augmentation
training_datagen = ImageDataGenerator(rescale=None,
                                      shear_range= 0.2,
                                      zoom_range= 0.2,
                                      horizontal_flip= True)

testing_datagen = ImageDataGenerator(rescale=1./255)

# Loading and augmenting training and testing datasets:

Training_set = training_datagen.flow_from_directory('Leafdata/train',
                                                    target_size= (128,128),
                                                    batch_size= 32,
                                                    class_mode='categorical')

labels = (Training_set.class_indices)
print(labels)

Test_set = testing_datagen.flow_from_directory('Leafdata/test',
                                                    target_size= (128,128),
                                                    batch_size= 32,
                                                    class_mode='categorical')

label = (Test_set.class_indices)
print(label)

model.fit_generator (Training_set,
                     steps_per_epoch= 375,
                     epochs= 10,
                     validation_data= Test_set,
                     validation_steps= 125)

model_json = model.to_json()
with open ( "Leafmodel.json" , w) as json_file:
    json_file.write(model_json)
    model.save_weights("LeafModel.h5")
    print("Model saved to disk")