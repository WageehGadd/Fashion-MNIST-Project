import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import gzip
import os
import pandas as pd
import cv2
import utils
from sklearn.model_selection import train_test_split

NUM_CLASSES = 10
SEED = 42

def get_data():
    train_df = pd.read_csv('../data/fashion-mnist_train.csv')
    test_df = pd.read_csv('../data/fashion-mnist_test.csv')

    y_train_raw = train_df['label'].values
    y_test_raw = test_df['label'].values

    x_train_raw = train_df.drop('label', axis=1).values
    x_test_raw = test_df.drop('label', axis=1).values

    x_train = x_train_raw.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test_raw.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    y_train_cat = to_categorical(y_train_raw, NUM_CLASSES)
    y_test_cat = to_categorical(y_test_raw, NUM_CLASSES)

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train,
        y_train_cat,
        test_size=0.10,
        random_state=SEED,
        stratify=y_train_raw
    )

    print("Data loaded from CSV and split.")
    print(f"Train: {x_tr.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    return x_tr, y_tr, x_val, y_val, x_test, y_test_cat

def get_generators(x_tr, y_tr, x_val, y_val):
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator()



    train_gen = train_datagen.flow(x_tr, y_tr, batch_size=utils.BATCH_SIZE, shuffle=True, seed=utils.SEED)
    val_gen = val_datagen.flow(x_val, y_val, batch_size=utils.BATCH_SIZE, shuffle=False)

    return train_gen, val_gen

def resize_and_stack(images, target_size=(85, 85)):
    processed_images = np.zeros(
        (images.shape[0], target_size[0], target_size[1], 3),
        dtype=np.float32
    )
    for i, img in enumerate(images):
        img_resized = cv2.resize(img, target_size)
        processed_images[i] = np.stack((img_resized,) * 3, axis=-1)
    return processed_images

get_data()