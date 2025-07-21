from pathlib import Path
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_confusion_matrix(model, data_loader, config):
    save_path = Path(config.callbacks.checkpoint_dir).parent

    y_true = [label for _, label in data_loader.get_test_data()]
    y_true = np.array([u for cl in y_true for u in cl]).argmax(axis=1)

    y_pred = np.argmax(model.predict(data_loader.get_test_data()), axis=-1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")

    # Save to image file (e.g., PNG)
    plt.savefig(
        save_path / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_training_history(trainer, config):
    save_path = Path(config.callbacks.checkpoint_dir).parent

    plt.figure(figsize=(12, 5))
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(trainer.loss, "bo-", label="Training Loss")
    plt.plot(trainer.val_loss, "ro-", label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(trainer.acc, "bo-", label="Training Accuracy")
    plt.plot(trainer.val_acc, "ro-", label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss(trainer, config):
    save_path = Path(config.callbacks.checkpoint_dir).parent

    plt.figure(figsize=(6, 4))
    plt.plot(trainer.loss, "bo-", label="Training Loss")
    plt.plot(trainer.val_loss, "ro-", label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path / "loss_over_epochs.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy(trainer, config):
    save_path = Path(config.callbacks.checkpoint_dir).parent

    plt.figure(figsize=(6, 4))
    plt.plot(trainer.acc, "bo-", label="Training Accuracy")
    plt.plot(trainer.val_acc, "ro-", label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path / "accuracy_over_epochs.png", dpi=300, bbox_inches="tight")

    plt.close()


def save_model_file(config):
    save_path = Path(config.callbacks.checkpoint_dir).parent
    print(Path(__file__).parent.parent / "models" / "blood_cell_model.py",)
    shutil.copy(
        Path(__file__).parent / "models" / "blood_cell_model.py",
        save_path / "blood_cell_model.py",
    )

def save_training_data(trainer, config):
    save_path = Path(config.callbacks.checkpoint_dir).parent
    with open(save_path / 'trainer.pkl', 'wb') as f:
        pickle.dump(trainer, f)
    