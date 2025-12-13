import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

import utils
from dataset import get_data, get_generators
from model import build_architecture


def main():
    print("Initializing Project Pipeline...")

    x_tr, y_tr, x_val, y_val, x_test, y_test = get_data()
    train_gen, val_gen = get_generators(x_tr, y_tr, x_val, y_val)

    print("Data loaded and generators initialized successfully.")

    model = build_architecture(utils.IMG_SHAPE, utils.NUM_CLASSES)
    model.summary()

    print("Model architecture built successfully.")
    print("System is ready for training module integration.")


if __name__ == "__main__":
    main()