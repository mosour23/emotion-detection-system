"""
=============================================================================
MODULE: lstm_classifier.py
PURPOSE: Deep Learning (LSTM/GRU) emotion classification model
DESCRIPTION:
    PyTorch-based LSTM and GRU models for emotion detection.
    Provides a scikit-learn compatible wrapper for integration with
    the existing training pipeline.
=============================================================================
"""

import logging
import numpy as np
from typing import Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available – LSTM/GRU models disabled")

logger = logging.getLogger(__name__)


class LSTMEmotionClassifier(nn.Module):
    """
    PyTorch LSTM model for emotion classification.
    
    Parameters
    ----------
    input_size : int
        Dimension of input features (e.g., TF-IDF dimension or embedding size)
    hidden_size : int
        Size of LSTM hidden state
    num_classes : int
        Number of emotion classes
    num_layers : int
        Number of LSTM layers
    dropout : float
        Dropout rate
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_classes: int = 6,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
            For static TF-IDF features, seq_len = 1
        
        Returns
        -------
        torch.Tensor
            Class logits of shape (batch_size, num_classes)
        """
        # LSTM output
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch_size, hidden_size)
        
        # FC layers
        logits = self.fc(last_hidden)
        return logits


class GRUEmotionClassifier(nn.Module):
    """
    PyTorch GRU model for emotion classification.
    
    Parameters
    ----------
    input_size : int
        Dimension of input features
    hidden_size : int
        Size of GRU hidden state
    num_classes : int
        Number of emotion classes
    num_layers : int
        Number of GRU layers
    dropout : float
        Dropout rate
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_classes: int = 6,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns
        -------
        torch.Tensor
            Class logits of shape (batch_size, num_classes)
        """
        gru_out, h_n = self.gru(x)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits


class PyTorchEmotionClassifier:
    """
    Scikit-learn compatible wrapper for PyTorch models.
    Handles training, prediction, and probability estimation.
    """
    
    def __init__(
        self,
        model_type: str = "lstm",
        input_size: int = 589,
        hidden_size: int = 128,
        num_classes: int = 6,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 20,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_type : str
            "lstm" or "gru"
        input_size : int
            Feature dimension
        hidden_size : int
            Hidden state size
        num_classes : int
            Number of emotion classes
        num_layers : int
            Number of RNN layers
        dropout : float
            Dropout rate
        learning_rate : float
            Optimizer learning rate
        epochs : int
            Number of training epochs
        batch_size : int
            Training batch size
        device : str
            "cuda" or "cpu"
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. Install with: pip install torch")
        
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        if model_type == "lstm":
            self.model = LSTMEmotionClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif model_type == "gru":
            self.model = GRUEmotionClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PyTorchEmotionClassifier":
        """
        Train the model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, D)
        y : np.ndarray
            Class labels (N,)
        
        Returns
        -------
        self
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Reshape for RNN: (batch, seq_len, features)
        # For static TF-IDF, treat entire feature vector as sequence of length 1
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(1)  # (N, 1, D)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                logger.info("Epoch %d/%d – Loss: %.4f", epoch + 1, self.epochs, total_loss / len(dataloader))
        
        self._fitted = True
        logger.info("%s model trained on %d samples", self.model_type.upper(), len(X))
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, D)
        
        Returns
        -------
        np.ndarray
            Predicted class labels (N,)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(1)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, D)
        
        Returns
        -------
        np.ndarray
            Class probabilities (N, num_classes)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(1)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()
