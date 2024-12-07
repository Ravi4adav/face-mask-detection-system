from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Defining model
mymodel=Sequential()
mymodel.add(Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)))
# Conv2D(2^n convolution layer, (row_in_convolution_layer, col_in_convolution_layer), activation_function, input_shape=(length,width, color))
# color=3 means rgb
mymodel.add(MaxPooling2D())

mymodel.add(Conv2D(32,(3,3), activation='relu'))
mymodel.add(MaxPooling2D())
mymodel.add(Conv2D(32,(3,3), activation='relu'))
mymodel.add(MaxPooling2D())

mymodel.add(Flatten())
mymodel.add(Dense(100, activation='relu'))
mymodel.add(Dense(1, activation='sigmoid'))
mymodel.compile(optimizer='Adam', loss='binary_crossentropy',metrics=['accuracy'])

# Defining Data
train=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test=ImageDataGenerator(rescale=1./255)
train_img=train.flow_from_directory('./Data/data/train',target_size=(150,150), batch_size=16,class_mode='binary')
test_img=test.flow_from_directory('./Data/data/test',target_size=(150,150), batch_size=16,class_mode='binary')

# Train and Test Model
mask_model=mymodel.fit(train_img, epochs=10, validation_data=(test_img))

# Save the model in my directory
mymodel.save('mask.h5',mask_model)