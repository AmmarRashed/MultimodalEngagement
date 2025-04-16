import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.notebook import tqdm


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.global_pool(x)
        return x.squeeze(-1)


class MultimodalFusion(nn.Module):
    def __init__(self, input_dims, hidden_dim=128, dropout=0.3):
        super().__init__()
        # input_dims should be a tuple of feature dimensions for each modality
        # e.g., (70, eye_features, pose_features)
        self.eeg_encoder = ModalityEncoder(input_dims[0], hidden_dim)
        self.eye_encoder = ModalityEncoder(input_dims[1], hidden_dim)
        self.pose_encoder = ModalityEncoder(input_dims[2], hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_tuple):
        eeg_feat = self.eeg_encoder(x_tuple[0])
        eye_feat = self.eye_encoder(x_tuple[1])
        pose_feat = self.pose_encoder(x_tuple[2])

        combined = torch.cat([eeg_feat, eye_feat, pose_feat], dim=1)
        return self.fusion(combined)


def train_model(model, train_loader, val_loader, device,
                best_model_path="best_model.pth",
                pos_weight=1,
                num_epochs=100, patience=10, learning_rate=1e-4):
    """
    Train the model with early stopping

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: torch device
        best_model_path: path to store the best model
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        learning_rate: Learning rate for optimizer
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in tqdm(range(num_epochs), leave=False, desc='Epochs'):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X, _, y in train_loader:
            # Move data to device
            X = tuple(x.to(device) for x in X)
            y = y.to(device).float()

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            predictions = (outputs > 0).float()
            train_correct += (predictions == y).sum().item()
            train_total += y.size(0)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, _, y in val_loader:
                X = tuple(x.to(device) for x in X)
                y = y.to(device).float()

                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                predictions = (outputs > 0).float()
                val_correct += (predictions == y).sum().item()
                val_total += y.size(0)

        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_train_acc = train_correct / train_total
        epoch_val_acc = val_correct / val_total

        # Store history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        # Print progress
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            print("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                break

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded best model from", best_model_path)

    return model, history


# Usage example:
"""
model = MultimodalFusion(input_dims=input_dims, hidden_dim=64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model, history = train_model(model, train_loader, val_loader, device)
"""


def get_predictions(model, test_loader, device):
    """
    Get model predictions on a test set.

    Args:
        model: The trained neural network model
        test_loader: DataLoader for test data
        device: torch device

    Returns:
        y_true: numpy array of true labels
        y_pred: numpy array of predicted labels (0/1)
        y_pred_prob: numpy array of prediction probabilities
    """
    model.eval()

    y_true_list = []
    y_pred_prob_list = []

    with torch.no_grad():
        for X, _, y in test_loader:
            # Move data to device
            X = tuple(x.to(device) for x in X)
            y = y.to(device)

            # Get model predictions
            outputs = model(X)
            probs = torch.sigmoid(outputs)

            # Store batch results
            y_true_list.append(y.cpu().numpy())
            y_pred_prob_list.append(probs.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(y_true_list)
    y_pred_prob = np.concatenate(y_pred_prob_list)

    # Get binary predictions using 0.5 threshold
    y_pred = (y_pred_prob >= 0.5).astype(np.int32)

    # Ensure all arrays are 1D
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    y_pred_prob = y_pred_prob.ravel()

    return y_true, y_pred, y_pred_prob


# Usage example:
"""
y_true, y_pred, y_pred_prob = get_predictions(model, test_loader, device)

# Now you can use these arrays for metrics:
from sklearn.metrics import classification_report, roc_auc_score

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nROC AUC Score:", roc_auc_score(y_true, y_pred_prob))
"""
