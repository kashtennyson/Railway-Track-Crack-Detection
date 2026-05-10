import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from .logger import ExperimentLogger
from . import config
from .data_loader import load_datasets

def evaluate():
    
    # Load data and model
    print("Loading test dataset and best model...")
    _, _, test_ds, class_names = load_datasets()
    model_path = os.path.join(config.OUTPUT_DIR, "best_model.keras")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the model first.")
        return

    model = tf.keras.models.load_model(model_path)

    # Get predictions
    y_true = []
    y_pred = []

    print("Generating predictions...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    # Integrate wandb experiment logger if enabled (evaluate)
    ExperimentLogger.log_evaluation(y_true, y_pred, class_names)

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Railway Track Crack Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save the plot
    plot_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"\nConfusion matrix saved to {plot_path}")

if __name__ == "__main__":
    evaluate()