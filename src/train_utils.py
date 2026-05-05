# Import os to create folders before saving model checkpoints.
import os

# Import numpy for calculating average loss and storing arrays.
import numpy as np

# Import torch for model training, prediction and tensor operations.
import torch

# Import tqdm to show progress bars during training and evaluation loops.
from tqdm import tqdm

# Import performance metrics for classification evaluation.
from sklearn.metrics import accuracy_score, f1_score


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    # Train the model for one complete pass through the training dataset.

    # Set the model to training mode.
    # This enables layers such as dropout and batch normalization to behave correctly during training.
    model.train()

    # Lists to store loss values, predictions and true labels.
    losses, all_preds, all_labels = [], [], []

    # Loop through batches of images and labels.
    for images, labels in tqdm(dataloader):

        # Move images and labels to the selected device: GPU or CPU.
        images, labels = images.to(device), labels.to(device)

        # Reset gradients from the previous batch.
        optimizer.zero_grad()

        # Forward pass: generate model outputs/logits.
        logits = model(images)

        # Calculate classification loss.
        loss = criterion(logits, labels)

        # Backward pass: calculate gradients.
        loss.backward()

        # Update model weights using the optimizer.
        optimizer.step()

        # Store the batch loss.
        losses.append(loss.item())

        # Convert logits into predicted class IDs.
        preds = torch.argmax(logits, dim=1)

        # Move predictions and labels to CPU and store them for metric calculation.
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    # Return average training loss and classification metrics for this epoch.
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted")
    }


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device):
    # Evaluate the model on validation or test data.
    # @torch.no_grad() disables gradient calculation, making evaluation faster and memory-efficient.

    # Set the model to evaluation mode.
    # This disables training-specific behaviour such as dropout.
    model.eval()

    # Lists to store loss values, predictions, labels and probability outputs.
    losses, all_preds, all_labels, all_probs = [], [], [], []

    # Loop through batches of images and labels.
    for images, labels in tqdm(dataloader):

        # Move images and labels to GPU or CPU.
        images, labels = images.to(device), labels.to(device)

        # Forward pass: generate model logits.
        logits = model(images)

        # Calculate validation/test loss.
        loss = criterion(logits, labels)

        # Convert logits into class probabilities using softmax.
        probs = torch.softmax(logits, dim=1)

        # Select the class with the highest probability.
        preds = torch.argmax(probs, dim=1)

        # Store the batch loss.
        losses.append(loss.item())

        # Store predictions, labels and probabilities on CPU for later analysis.
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    # Return loss, metrics and raw prediction arrays.
    # labels, preds and probs are useful for confusion matrices and fusion models.
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted"),
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs)
    }


def fit_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, checkpoint_path):
    # Train the model for multiple epochs and save the best checkpoint.

    # Track the best validation macro F1-score seen so far.
    # Macro F1 is useful because it treats all sentiment classes equally.
    best_f1 = -1

    # Store training and validation history for every epoch.
    history = []

    # Loop through each training epoch.
    for epoch in range(1, epochs + 1):

        # Print current epoch number.
        print(f"\nEpoch {epoch}/{epochs}")

        # Train the model for one epoch.
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # Evaluate the model on the validation set.
        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            device
        )

        # Store important metrics for this epoch.
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"]
        }

        # Add epoch results to the training history.
        history.append(row)

        # Print epoch results in the terminal.
        print(row)

        # Save the model if it has the best validation macro F1 so far.
        if val_metrics["macro_f1"] > best_f1:
            # Update the best F1-score.
            best_f1 = val_metrics["macro_f1"]

            # Create checkpoint folder if it does not exist.
            os.makedirs(os.path.dirname(str(checkpoint_path)), exist_ok=True)

            # Save model weights to the checkpoint path.
            torch.save(model.state_dict(), checkpoint_path)

            # Print confirmation message.
            print("Saved best checkpoint:", checkpoint_path)

    # Return the full training history.
    return history