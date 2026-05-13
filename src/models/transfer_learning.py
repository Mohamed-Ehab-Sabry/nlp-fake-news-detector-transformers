"""
Transfer Learning Models for Fake News Detection

Pre-trained Embeddings:
- GloVe (Global Vectors for Word Representation)
- FastText (Facebook's fastText)
- Word2Vec (Skip-gram model)

Approaches:
- Fine-tuning pre-trained embeddings
- Frozen embeddings for efficiency
- LSTM/GRU with transfer learning
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path


class TransferLearningModel:
    """Transfer Learning with pre-trained embeddings"""

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        max_length: int = 150,
        trainable_embeddings: bool = False,
        lstm_units: int = 128,
    ):
        """
        Initialize transfer learning model

        Args:
            embedding_matrix: Pre-trained embedding matrix (vocab_size, embedding_dim)
            max_length: Maximum sequence length
            trainable_embeddings: Whether to fine-tune embeddings
            lstm_units: Number of LSTM units
        """
        self.embedding_matrix = embedding_matrix
        self.max_length = max_length
        self.trainable_embeddings = trainable_embeddings
        self.lstm_units = lstm_units
        self.model = None

    def build(self, num_classes: int = 1, use_bidirectional: bool = True):
        """Build transfer learning model"""
        vocab_size, embedding_dim = self.embedding_matrix.shape

        input_layer = layers.Input(shape=(self.max_length,), dtype="int32")

        embedding = layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=self.max_length,
            weights=[self.embedding_matrix],
            trainable=self.trainable_embeddings,
        )(input_layer)

        if use_bidirectional:
            lstm = layers.Bidirectional(
                layers.LSTM(self.lstm_units, return_sequences=True)
            )(embedding)
        else:
            lstm = layers.LSTM(self.lstm_units, return_sequences=True)(embedding)

        # Attention mechanism
        attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)(lstm, lstm)
        pool = layers.GlobalAveragePooling1D()(attention)

        dense1 = layers.Dense(64, activation="relu")(pool)
        dropout = layers.Dropout(0.5)(dense1)
        output = layers.Dense(num_classes, activation="sigmoid")(dropout)

        self.model = Model(inputs=input_layer, outputs=output)
        return self.model

    def compile(self, optimizer: str = "adam", loss: str = "binary_crossentropy"):
        """Compile model"""
        if self.trainable_embeddings:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

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
                f"{model_dir}/transfer_learning_best.keras",
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


class EmbeddingManager:
    """Manager for pre-trained embeddings"""

    def __init__(self):
        """Initialize embedding manager"""
        self.embeddings = {}

    def load_glove(
        self, filepath: str, embedding_dim: int = 300
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load GloVe embeddings

        Args:
            filepath: Path to GloVe file
            embedding_dim: Embedding dimension

        Returns:
            Embedding matrix and word index dictionary
        """
        embeddings_index = {}
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

        self.embeddings["glove"] = embeddings_index
        return embeddings_index

    def load_fasttext(
        self, filepath: str, embedding_dim: int = 300
    ) -> Tuple[np.ndarray, Dict]:
        """Load FastText embeddings"""
        embeddings_index = {}
        try:
            import fasttext

            model = fasttext.load_model(filepath)
            embeddings_index = model
        except ImportError:
            raise ImportError("Please install fasttext: pip install fasttext")

        self.embeddings["fasttext"] = embeddings_index
        return embeddings_index

    def get_embedding_matrix(
        self, word_index: Dict, embedding_source: str, embedding_dim: int
    ) -> np.ndarray:
        """
        Create embedding matrix for vocabulary

        Args:
            word_index: Dictionary mapping words to indices
            embedding_source: Source of embeddings (glove, fasttext, word2vec)
            embedding_dim: Embedding dimension

        Returns:
            Embedding matrix (vocab_size, embedding_dim)
        """
        vocab_size = max(word_index.values()) + 1
        embedding_matrix = np.random.normal(size=(vocab_size, embedding_dim))

        embeddings_index = self.embeddings.get(embedding_source, {})

        for word, idx in word_index.items():
            if word in embeddings_index:
                embedding_vector = embeddings_index[word]
                if embedding_vector is not None:
                    embedding_matrix[idx] = embedding_vector

        return embedding_matrix


def get_pretrained_embedding(
    embedding_type: str = "glove", embedding_dim: int = 300
) -> np.ndarray:
    """
    Get pre-trained embedding matrix

    Args:
        embedding_type: Type of embedding (glove, fasttext, word2vec)
        embedding_dim: Embedding dimension

    Returns:
        Embedding matrix
    """
    model_dir = Path("results/models/embeddings")

    if embedding_type == "glove":
        filepath = model_dir / f"glove.6B.{embedding_dim}d.txt"
    elif embedding_type == "fasttext":
        filepath = model_dir / "fasttext.bin"
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    if not filepath.exists():
        raise FileNotFoundError(f"Embedding file not found: {filepath}")

    manager = EmbeddingManager()
    if embedding_type == "glove":
        embeddings_index = manager.load_glove(str(filepath), embedding_dim)
    elif embedding_type == "fasttext":
        embeddings_index = manager.load_fasttext(str(filepath), embedding_dim)

    return embeddings_index
