import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

import utils
from dataset import get_data, get_generators
from model import build_architecture
from train import run_training

def main():
    print("Initializing Project Pipeline...")

    print("Loading and preparing data...")
    x_tr, y_tr, x_val, y_val, x_test, y_test = get_data()
    train_gen, val_gen = get_generators(x_tr, y_tr, x_val, y_val)
    print("Data loaded and generators initialized.")

    print("Verifying model architecture...")
    model_arch = build_architecture(utils.IMG_SHAPE, utils.NUM_CLASSES)
    model_arch.summary()

    print("Starting training module...")
    model, history = run_training(train_gen, val_gen)

    print("Training process completed.")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main()