import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import utils
from dataset import get_data
import pickle


import os
if not os.path.exists(utils.RESULTS_DIR):
    os.makedirs(utils.RESULTS_DIR)

def plot_training_history_1(history):
    """
    Function to plot and save training curves (Loss & Accuracy) - Basic Version
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 5))

    # 1. Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    # 2. Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Save with different filename
    plot_path = os.path.join(utils.RESULTS_DIR, 'training_curves_1.png')
    plt.savefig(plot_path)
    print(f"Basic training curves saved to {plot_path}")
    plt.close()

def plot_training_histor_2(history):
    """
    Function to plot and save enhanced training curves (Loss & Accuracy)
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Plot Accuracy
    axes[0].plot(acc, label='Training Accuracy',
                 linewidth=2, marker='o', markersize=4)
    axes[0].plot(val_acc, label='Validation Accuracy',
                 linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Accuracy over Epochs', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Highlight best validation accuracy
    best_val_acc = max(val_acc)
    best_epoch = val_acc.index(best_val_acc)
    axes[0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
    axes[0].text(best_epoch, best_val_acc, f'Best: {best_val_acc:.4f}',
                 fontsize=9, ha='right')

    # Plot Loss
    axes[1].plot(loss, label='Training Loss',
                 linewidth=2, marker='o', markersize=4)
    axes[1].plot(val_loss, label='Validation Loss',
                 linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Loss over Epochs', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11, loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save with different filename
    plot_path = os.path.join(utils.RESULTS_DIR, 'training_curves_2.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"another training curves saved to {plot_path}")
    
    # Print summary statistics
    print(f"\nFinal Training Accuracy: {acc[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_acc[-1]:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")
    
    plt.close()


def plot_separate_curves(history):
    """
    Function to plot and save separate accuracy and loss curves
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Find best validation accuracy
    best_val_acc = max(val_acc)
    best_epochs = [i for i, acc_val in enumerate(val_acc) if acc_val == best_val_acc]
    best_epoch = best_epochs[0]

    # ACCURACY CURVE 
    plt.figure(figsize=(10, 6))
    plt.plot(acc, label='Training Accuracy',
             linewidth=2, marker='o', markersize=5, color='#2E86AB')
    plt.plot(val_acc, label='Validation Accuracy',
             linewidth=2, marker='s', markersize=5, color='#A23B72')
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
    plt.title('Model Accuracy over Epochs', fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Highlight best validation accuracy
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(best_epoch, best_val_acc, f'  Best: {best_val_acc:.4f}\n  Epoch {best_epoch + 1}',
             fontsize=10, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    accuracy_path = os.path.join(utils.RESULTS_DIR, 'accuracy_curve.png')
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy curve saved to {accuracy_path}")
    plt.close()

    # LOSS CURVE 

    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss',
             linewidth=2, marker='o', markersize=5, color='#F18F01')
    plt.plot(val_loss, label='Validation Loss',
             linewidth=2, marker='s', markersize=5, color='#C73E1D')
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Loss', fontsize=13, fontweight='bold')
    plt.title('Model Loss over Epochs', fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Highlight best epoch
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, linewidth=2)
    best_val_loss = val_loss[best_epoch]
    plt.text(best_epoch, best_val_loss, f'  Best Epoch: {best_epoch + 1}\n  Loss: {best_val_loss:.4f}',
             fontsize=10, ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    loss_path = os.path.join(utils.RESULTS_DIR, 'loss_curve.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve saved to {loss_path}")
    plt.close()


def analyze_sample_predictions(model, x_test, y_test, sample_size=20, class_names=None):
    """
    Analyze and display individual predictions from the model.
    Save results to a text file.
    """
    sample_preds = model.predict(x_test[:sample_size], verbose=0)
    
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_labels = np.argmax(y_test[:sample_size], axis=1)
    else:
        y_test_labels = y_test[:sample_size]

    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("SAMPLE PREDICTIONS ANALYSIS")
    output_lines.append("=" * 70)
    output_lines.append(f"\nFirst {sample_size} test samples:")
    output_lines.append(f"{'Sample':<8} {'True':<15} {'Pred':<15} {'Confidence':<12} {'Match':<6}")
    output_lines.append("-" * 70)

    matches = 0
    for i in range(sample_size):
        true_label = y_test_labels[i]
        pred_label = np.argmax(sample_preds[i])
        confidence = sample_preds[i].max()
        match = "PASS" if true_label == pred_label else "FAIL"
        
        if true_label == pred_label:
            matches += 1

        if class_names:
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
            line = f"{i:<8} {true_name:<15} {pred_name:<15} {confidence:<12.4f} {match:<6}"
        else:
            line = f"{i:<8} {true_label:<15} {pred_label:<15} {confidence:<12.4f} {match:<6}"
        
        output_lines.append(line)

    accuracy = matches / sample_size * 100
    output_lines.append(f"\nAccuracy on these {sample_size} samples: {matches}/{sample_size} = {accuracy:.1f}%")

    pred_labels = np.argmax(sample_preds, axis=1)
    pred_distribution = np.bincount(pred_labels, minlength=len(class_names) if class_names else 10)
    
    output_lines.append(f"\nPrediction distribution across classes:")
    for class_idx, count in enumerate(pred_distribution):
        if class_names and class_idx < len(class_names):
            output_lines.append(f"  Class {class_idx} ({class_names[class_idx]}): {count} predictions")
        else:
            output_lines.append(f"  Class {class_idx}: {count} predictions")
    
    # Save to file with UTF-8 encoding
    output_path = os.path.join(utils.RESULTS_DIR, 'sample_predictions_analysis.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\nSample predictions analysis saved to {output_path}")
    

    return {
        'sample_size': sample_size,
        'matches': matches,
        'accuracy': accuracy,
        'predictions': sample_preds,
        'pred_labels': pred_labels,
        'true_labels': y_test_labels,
        'pred_distribution': pred_distribution
    }



def evaluate_model():
    """
    Function to evaluate the saved model on test data
    """
    history_path = os.path.join(utils.MODEL_DIR, 'training_history.pkl')
    if os.path.exists(history_path):
        print(f"Loading training history from: {history_path}")
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        # Generate  curves
        plot_training_history_1(history)
        plot_training_histor_2(history)
        plot_separate_curves(history)

    else:
        print(f"Training history not found at {history_path}. Skipping training curves plot.")
    
    print("Loading test data...")
    _, _, _, _, x_test, y_test_cat = get_data()
    
    test_datagen = ImageDataGenerator()
    test_gen = test_datagen.flow(x_test, y_test_cat, batch_size=utils.BATCH_SIZE, shuffle=False)

    model_path = os.path.join(utils.MODEL_DIR, 'best_model.h5')
    
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return

    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    print("Running evaluation on test set...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    analyze_sample_predictions(
        model=model,
        x_test=x_test,
        y_test=y_test_cat,
        sample_size=20,
        class_names=utils.CLASS_NAMES
    )
 
    report = classification_report(y_true, y_pred, target_names=utils.CLASS_NAMES)
    report_path = os.path.join(utils.RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")


    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=utils.CLASS_NAMES, yticklabels=utils.CLASS_NAMES, cmap='Blues')
    plt.title(f'Confusion Matrix (Acc: {test_acc:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(utils.RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion Matrix saved to {cm_path}")
    plt.close()


if __name__ == "__main__":
    evaluate_model()