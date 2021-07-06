import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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
X_train_prep, y_test, _ = load_data(TEST_DIR, IMG_SIZE)
X_val_prep, y_val, _ = load_data(VAL_DIR, IMG_SIZE)







	
