import torch
import torch.nn.functional as F
import numpy as np
import random, os
import time
from collections import defaultdict
from typing import Tuple, Dict, Any, List
from warnings import simplefilter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from .metrics import get_eval_metrics
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim

# ANN Binary Classifier (Now Outputs Two Class Probabilities)
class ANNBinaryClassifier:
    def __init__(self, input_dim=512, hidden_dim=512, max_iter=100, C=1.0, verbose=True):
        self.C = C
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        #Define the model (Two outputs for [MSS, MSI])
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, 2),  #   Now outputs [P(MSS), P(MSI)]
            nn.Softmax(dim=1)  #   Ensure probabilities sum to 1
        ).to(self.device)

        #   Use CrossEntropyLoss (since we now have two outputs)
        self.loss_func = nn.CrossEntropyLoss()

    def compute_loss(self, preds, labels):
        return self.loss_func(preds, labels)

    def predict_proba(self, feats):
        feats = feats.to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(feats)  #   Now returns two probabilities per sample

    def fit(self, train_feats, train_labels, val_feats=None, val_labels=None, combine_trainval=False):
        train_feats, train_labels = train_feats.to(self.device), train_labels.to(self.device)
        if val_feats is not None:
            val_feats, val_labels = val_feats.to(self.device), val_labels.to(self.device)
        if combine_trainval and val_feats is not None:
            train_feats = torch.cat([train_feats, val_feats], dim=0)
            train_labels = torch.cat([train_labels, val_labels], dim=0)

        # opt = optim.Adam(self.model.parameters(), lr=1e-4)
        # scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)  #   Using original StepLR
        opt = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.3, patience=5, verbose=True)

        train_loss_history, val_loss_history = [], []
        best_val_loss = float("inf")
        patience, epochs_no_improve = 10, 0

        for epoch in range(self.max_iter):
            self.model.train()
            preds = self.model(train_feats)
            loss = self.compute_loss(preds, train_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_history.append(loss.item())

            # Validation phase
            val_loss = None
            if val_feats is not None and not combine_trainval:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(val_feats)
                    val_loss = self.compute_loss(val_preds, val_labels)

                val_loss_history.append(val_loss.item())

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break

            #   Keep StepLR scheduler
            if val_loss is not None:
                scheduler.step(val_loss)  # Call with val_loss if available
            else:
                scheduler.step(loss)  # Call without val_loss when validation data is missing

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss: {loss:.3f}, Val Loss: {val_loss:.3f}" if val_loss else f"Epoch {epoch}: Loss: {loss:.3f}")

        return train_loss_history, val_loss_history

def eval_ANN(
    fold: int,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    valid_feats: torch.Tensor,
    valid_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    input_dim: int = 512,
    hidden_dim: int = 512,
    max_iter: int = 1000,
    prefix: str = "ann_",
    combine_trainval: bool = False,
    model_save_path: str = None,
    verbose: bool = False,
    num_runs: int = 1,  # Number of runs
    runs_results: str = "average",  # Choose between "average" or "best"
    metric_weights: Dict[str, float] = None,  # Weights for metrics (e.g., {"balanced_accuracy": 0.5, "aucroc": 0.5})
) -> tuple:
    if verbose:
        print(f"Train Shape: {train_feats.shape}, Test Shape: {test_feats.shape}")

    best_eval_metrics = None
    best_model_state = None
    best_run_index = -1
    all_eval_metrics = []
    best_dump = None  # To store the dump for the best run

    # Default metric weights if not provided
    if metric_weights is None:
        metric_weights = {"balanced_accuracy": 0.5, "aucroc": 0.5}

    def calculate_composite_score(metrics):
        """Calculate a composite score based on the provided metric weights."""
        return sum(metrics[f"{prefix}{metric}"] * weight for metric, weight in metric_weights.items())

    for run in range(num_runs):
        if verbose:
            print(f"Run {run + 1}/{num_runs}")

        classifier = ANNBinaryClassifier(input_dim=input_dim, hidden_dim=hidden_dim, max_iter=max_iter, verbose=verbose)
        train_loss, val_loss = classifier.fit(train_feats, train_labels, valid_feats, valid_labels, combine_trainval)

        # Testing phase
        probs_all = classifier.predict_proba(test_feats).cpu().numpy()
        preds_all = np.argmax(probs_all, axis=1)  # Predict class labels
        targets_all = test_labels.cpu().numpy()

        # Compute evaluation metrics
        eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, prefix=prefix)
        all_eval_metrics.append(eval_metrics)

        # Calculate composite score for the current run
        composite_score = calculate_composite_score(eval_metrics)

        # Save the best model based on the composite score
        if best_eval_metrics is None or composite_score > calculate_composite_score(best_eval_metrics):
            best_eval_metrics = eval_metrics
            best_model_state = classifier.model.state_dict()
            best_run_index = run
            best_dump = {"preds_all": preds_all, "probs_all": probs_all, "targets_all": targets_all}

        if verbose:
            print(f"Run {run + 1} Metrics: {eval_metrics}")
            print(f"Run {run + 1} Composite Score: {composite_score}")

    # Save the best model weights
    if model_save_path is not None and best_model_state is not None:
        model_path = os.path.join(model_save_path, f"fold{fold}_best_ann_model_{input_dim}.pth")
        torch.save(best_model_state, model_path)
        if verbose:
            print(f"Best model saved at: {model_path}")

    # Compute average metrics across all runs
    avg_eval_metrics = defaultdict(float)
    for metrics in all_eval_metrics:
        for key, value in metrics.items():
            avg_eval_metrics[key] += value
    avg_eval_metrics = {key: value / num_runs for key, value in avg_eval_metrics.items()}

    if verbose:
        print(f"Best Run Index: {best_run_index + 1}")
        print(f"Best Metrics: {best_eval_metrics}")
        print(f"Average Metrics: {avg_eval_metrics}")
        plot_training_logs({"train_loss": train_loss, "valid_loss": val_loss})
        plot_roc_auc(targets_all, probs_all[:, 1])
    # Return results based on the `runs_results` parameter
    if runs_results == "average":
        # print best_eval_metrics and avg_eval_metrics
        return avg_eval_metrics, best_dump
    else:
        return best_eval_metrics, best_dump

#   Function to Load and Test Saved ANN Model
def test_saved_ann_model(input_dim: int, hidden_dim: int, test_feats: torch.Tensor, test_labels: torch.Tensor, model_path="best_ann_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #   Define model structure (Must match trained model)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim),
        nn.Dropout(0.5),
        nn.Linear(hidden_dim, 2),  #   Two outputs for [MSS, MSI]
        nn.Softmax(dim=1)  #   Ensure outputs sum to 1
    ).to(device)

    #   Load trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    #   Convert test features to tensor
    test_feats = test_feats.to(device)

    #   Get predictions
    with torch.no_grad():
        probabilities = model(test_feats).cpu().numpy()

    #   Convert probabilities to class labels (binary classification)
    predictions = np.argmax(probabilities, axis=1)
    targets_all = test_labels.cpu().numpy()

    #   Compute evaluation metrics
    eval_metrics = get_eval_metrics(targets_all, predictions, probabilities,True, prefix="ann_")
    dump = {"preds_all": predictions, "probs_all": probabilities, "targets_all": targets_all}
    return eval_metrics, dump


def plot_training_logs(training_logs):
    plt.figure(figsize=(10, 6))
    plt.plot(training_logs["train_loss"], label="Train Loss", marker="o")
    if "valid_loss" in training_logs and training_logs["valid_loss"]:
        plt.plot(training_logs["valid_loss"], label="Validation Loss", marker="x")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roc_auc(targets, probs):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
