"""
Deep Learning Models for Fake News Detection

Models:
- Convolutional Neural Networks (CNN)
- Autoencoders
- Bidirectional LSTM with Attention

Features:
- Word embedding layers
- Convolutional filters for n-gram patterns
- GlobalMaxPooling for feature extraction
- Early stopping and model checkpointing
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path


class CNN:
    """Convolutional Neural Network for text classification"""

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        max_length: int = 150,
        filters: list = None,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize CNN model

        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            max_length: Maximum sequence length
            filters: List of filter sizes (e.g., [3, 4, 5])
            dropout_rate: Dropout rate
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.filters = filters or [3, 4, 5]
        self.dropout_rate = dropout_rate
        self.model = None

    def build(self, num_classes: int = 1):
        """Build CNN model"""
        input_layer = layers.Input(shape=(self.max_length,), dtype="int32")

        embedding = layers.Embedding(
            self.vocab_size, self.embedding_dim, input_length=self.max_length
        )(input_layer)

        conv_blocks = []
        for filter_size in self.filters:
            conv = layers.Conv1D(
                100, filter_size, activation="relu", padding="valid"
            )(embedding)
            pool = layers.GlobalMaxPooling1D()(conv)
            conv_blocks.append(pool)

        concatenated = (
            layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        )
        dense1 = layers.Dense(256, activation="relu")(concatenated)
        dropout1 = layers.Dropout(self.dropout_rate)(dense1)
        dense2 = layers.Dense(64, activation="relu")(dropout1)
        dropout2 = layers.Dropout(self.dropout_rate * 0.6)(dense2)

        output = layers.Dense(num_classes, activation="sigmoid")(dropout2)

        self.model = Model(inputs=input_layer, outputs=output)
        return self.model

    def compile(self, optimizer: str = "adam", loss: str = "binary_crossentropy"):
        """Compile model"""
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        model_dir: str = "results/models",
    ) -> Dict:
        """
        Train model

        Returns:
            Training history
        """
        checkpoint = ModelCheckpoint(
            f"{model_dir}/cnn_best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stop],
            verbose=1,
        )

        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

    def save(self, filepath: str):
        """Save model"""
        self.model.save(filepath)

    def load(self, filepath: str):
        """Load model"""
        self.model = models.load_model(filepath)


class Autoencoder:
    """Autoencoder for anomaly detection"""

    def __init__(
        self,
        input_dim: int = 150 * 128,
        encoder_dims: list = None,
        latent_dim: int = 64,
    ):
        """
        Initialize Autoencoder

        Args:
            input_dim: Input dimension
            encoder_dims: List of encoder layer dimensions
            latent_dim: Latent space dimension
        """
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims or [256, 128]
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None

    def build(self):
        """Build autoencoder"""
        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,))
        x = input_layer

        for dim in self.encoder_dims:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(0.3)(x)

        latent = layers.Dense(self.latent_dim, activation="relu", name="latent")(x)

        # Decoder
        x = latent
        for dim in reversed(self.encoder_dims):
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(0.3)(x)

        output = layers.Dense(self.input_dim, activation="sigmoid")(x)

        self.model = Model(inputs=input_layer, outputs=output)
        self.encoder = Model(inputs=input_layer, outputs=latent)

        return self.model

    def compile(self, optimizer: str = "adam", loss: str = "mse"):
        """Compile model"""
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])

    def train(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray = None,
        epochs: int = 30,
        batch_size: int = 32,
    ) -> Dict:
        """Train autoencoder"""
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        ]

        history = self.model.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        return history.history

    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Calculate reconstruction error"""
        predictions = self.model.predict(X)
        mse = np.mean(np.square(X - predictions), axis=1)
        return mse

    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """Get latent representation"""
        return self.encoder.predict(X)

    def save(self, filepath: str):
        """Save model"""
        self.model.save(filepath)

    def load(self, filepath: str):
        """Load model"""
        self.model = models.load_model(filepath)
        self.encoder = Model(inputs=self.model.input, outputs=self.model.get_layer("latent").output)


class BidirectionalLSTM:
    """Bidirectional LSTM with optional attention"""

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        max_length: int = 150,
        lstm_units: int = 128,
        use_attention: bool = False,
        dropout_rate: float = 0.5,
    ):
        """Initialize Bidirectional LSTM"""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.lstm_units = lstm_units
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        self.model = None

    def build(self, num_classes: int = 1):
        """Build model"""
        input_layer = layers.Input(shape=(self.max_length,), dtype="int32")

        embedding = layers.Embedding(
            self.vocab_size, self.embedding_dim, input_length=self.max_length
        )(input_layer)

        lstm = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=self.use_attention)
        )(embedding)

        if self.use_attention:
            attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)(
                lstm, lstm
            )
            lstm = layers.GlobalAveragePooling1D()(attention)
        else:
            lstm = layers.GlobalAveragePooling1D()(lstm)

        dense1 = layers.Dense(64, activation="relu")(lstm)
        dropout1 = layers.Dropout(self.dropout_rate)(dense1)
        output = layers.Dense(num_classes, activation="sigmoid")(dropout1)

        self.model = Model(inputs=input_layer, outputs=output)
        return self.model

    def compile(self, optimizer: str = "adam", loss: str = "binary_crossentropy"):
        """Compile model"""
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        model_dir: str = "results/models",
    ) -> Dict:
        """Train model"""
        callbacks = [
            ModelCheckpoint(
                f"{model_dir}/lstm_best.keras",
                monitor="val_loss",
                save_best_only=True,
            ),
            EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
        ]

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

    def save(self, filepath: str):
        """Save model"""
        self.model.save(filepath)

    def load(self, filepath: str):
        """Load model"""
        self.model = models.load_model(filepath)


def get_dl_model(model_name: str, model_dir: str = "results/models") -> Model:
    """
    Load pre-trained deep learning model

    Args:
        model_name: Name of model file
        model_dir: Directory containing models

    Returns:
        Loaded model
    """
    model_path = Path(model_dir) / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return models.load_model(model_path)
