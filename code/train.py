import os
import sys
from tensorflow.keras import optimizers, callbacks

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils
import model as model_lib


def run_training(train_gen, val_gen):
    model = model_lib.build_architecture(utils.IMG_SHAPE, utils.NUM_CLASSES)

    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_path = os.path.join(utils.MODEL_DIR, 'best_model.h5')

    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=utils.PATIENCE, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=utils.EPOCHS,
        callbacks=cb_list,
        verbose=1
    )

    return model, history