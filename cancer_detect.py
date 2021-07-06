import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# load data function
def load_data(dir_path,img_size=(100,100)):
	X = []
	y = []
	i = 0
	labels = dict()
	for path in sorted(os.listdir(dir_path)):
		if not path.startswith('.'):
			labels[i] = path
			for file in os.listdir(dir_path + path):
				if not file.startswith('.'):
					img = cv2.imread(dir_path + path +'/' + file)
					X.append(img)
					y.append(i)
	X = np.array(X)
	y = np.array(y)
	print(f'{len(X)} images loaded from {dir_path} directory')
	return X,y, labels

TRAIN_DIR = 'TRAIN_CROP/'
TEST_DIR = 'TEST_CROP/'
VAL_DIR = 'VAL_CROP/'
IMG_SIZE = (224,224)


X_train_prep, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)
X_test_prep, y_test, _ = load_data(TEST_DIR, IMG_SIZE)
X_val_prep, y_val, _ = load_data(VAL_DIR, IMG_SIZE)

# preprocessing the data before training
def preprocess_imgs(inp, img_size):
	new = []
	for img in inp:
		img = cv2.resize(
			img,
			dsize = img_size,
			interpolation = cv2.INTER_CUBIC
		)

		new.append(tf.keras.applications.vgg16.preprocess_input(img))
	return np.array(new)

X_train_prep = preprocess_imgs(X_train_prep, img_size = IMG_SIZE)
X_test_prep = preprocess_imgs(X_test_prep, img_size = IMG_SIZE)
X_val_prep = preprocess_imgs(X_val_prep, img_size = IMG_SIZE)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    seed=123
)


validation_generator = test_datagen.flow_from_directory(
    VAL_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary',
    seed=123
)

# model building
base_model = tf.keras.applications.mobilenet.MobileNet(
	input_shape = IMG_SIZE + (3,),
	include_top=False, weights = 'imagenet', input_tensor=None,
	pooling = None
)

model = tf.keras.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.layers[0].trainable = False

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer = opt, metrics=['accuracy'])

model.summary()

# epochs summary
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc', 
    mode='max',
    patience=6
)
# increase epochs and other steps for better training
history = model.fit_generator(
    train_generator,
    steps_per_epoch=3,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=1,
    callbacks=[es]
)

# vallidation on val set of data
pred = model.predict(X_val_prep)
pred = [1 if x>0.5 else 0 for x in pred]

accuracy = accuracy_score(y_val, pred)
print('Val Accuracy = %.2f' % accuracy)

confusion_matrix(y_val, pred) 
