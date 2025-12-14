
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import  optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from dataset import *

x_train , y_train , x_val , y_val , x_test , y_test = get_data()
x_train_processed = resize_and_stack(x_train)
x_val_processed = resize_and_stack(x_val)
x_test_processed = resize_and_stack(x_test)

x_train_processed , x_val_processed = get_generators(x_train_processed, y_train,  x_val_processed ,y_val)


base_model = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(85, 85, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

lr_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    factor=0.5,
    min_lr=1e-5,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train_processed,y_train,
    epochs=10,
    validation_data=(x_val_processed, y_val),
    callbacks=[lr_reduction]
)

base_model.trainable = True

model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train_processed, y_train,
    epochs=20,
    validation_data=(x_val_processed, y_val),
    callbacks=[lr_reduction, early_stop]
)

test_loss, test_acc = model.evaluate(x_test_processed, y_test)
print(test_acc)
