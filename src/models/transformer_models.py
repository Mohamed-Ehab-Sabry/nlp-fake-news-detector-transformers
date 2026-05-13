"""
Transformer-based Models for Fake News Detection

Models:
- BERT fine-tuning (DistilBERT, RoBERTa compatible)
- Ensemble methods combining multiple models

Features:
- HuggingFace transformers integration
- Tokenization and preprocessing
- Fine-tuning strategies
- Ensemble voting
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path

try:
    from transformers import (
        TFBertModel,
        BertTokenizer,
        TFBertForSequenceClassification,
        AutoTokenizer,
        TFAutoModelForSequenceClassification,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BertClassifier:
    """BERT-based text classifier"""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 150,
        num_classes: int = 2,
    ):
        """
        Initialize BERT classifier

        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
            num_classes: Number of classes
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library required: pip install transformers"
            )

        self.model_name = model_name
        self.max_length = max_length
        self.num_classes = num_classes

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

    def encode_texts(self, texts: List[str]) -> Dict:
        """
        Encode texts using BERT tokenizer

        Args:
            texts: List of text strings

        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )

        return encoded

    def prepare_dataset(
        self, texts: List[str], labels: np.ndarray, batch_size: int = 16
    ) -> tf.data.Dataset:
        """
        Prepare dataset for training

        Args:
            texts: List of texts
            labels: Array of labels
            batch_size: Batch size

        Returns:
            TensorFlow dataset
        """
        encoded = self.encode_texts(texts)

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "token_type_ids": encoded["token_type_ids"],
                },
                labels,
            )
        )

        dataset = dataset.shuffle(buffer_size=len(texts))
        dataset = dataset.batch(batch_size)

        return dataset

    def compile(self, learning_rate: float = 2e-5):
        """Compile model"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"],
        )

    def train(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: List[str],
        val_labels: np.ndarray,
        epochs: int = 3,
        batch_size: int = 16,
        model_dir: str = "results/models",
    ) -> Dict:
        """
        Train BERT model

        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            model_dir: Directory to save models

        Returns:
            Training history
        """
        train_dataset = self.prepare_dataset(train_texts, train_labels, batch_size)
        val_dataset = self.prepare_dataset(val_texts, val_labels, batch_size)

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            verbose=1,
        )

        # Save model
        model_path = Path(model_dir) / "bert_model"
        self.model.save_pretrained(model_path)

        return history.history

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions

        Args:
            texts: List of texts

        Returns:
            Tuple of (predictions, probabilities)
        """
        encoded = self.encode_texts(texts)
        outputs = self.model(encoded)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs[0]

        predictions = tf.argmax(logits, axis=1).numpy()
        probabilities = tf.nn.softmax(logits, axis=1).numpy()

        return predictions, probabilities

    def save(self, filepath: str):
        """Save model"""
        self.model.save_pretrained(filepath)

    def load(self, filepath: str):
        """Load model"""
        self.model = TFAutoModelForSequenceClassification.from_pretrained(filepath)


class EnsembleModel:
    """Ensemble of multiple models with soft voting"""

    def __init__(self, models_dict: Dict[str, Any], weights: Dict[str, float] = None):
        """
        Initialize ensemble

        Args:
            models_dict: Dictionary of {model_name: model_object}
            weights: Dictionary of {model_name: weight}
        """
        self.models = models_dict
        self.weights = weights or {name: 1.0 / len(models_dict) for name in models_dict}

    def predict(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions using soft voting

        Args:
            X: Input data

        Returns:
            Tuple of (class_predictions, probabilities)
        """
        predictions_list = []

        for model_name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
            elif callable(model.predict):
                output = model.predict(X)
                if isinstance(output, tuple):
                    _, probs = output
                else:
                    probs = output

            predictions_list.append(probs)

        # Weighted averaging
        ensemble_probs = np.zeros_like(predictions_list[0], dtype=np.float32)

        for i, model_name in enumerate(self.models.keys()):
            weight = self.weights.get(model_name, 1.0 / len(self.models))
            ensemble_probs += weight * predictions_list[i]

        ensemble_predictions = np.argmax(ensemble_probs, axis=1)

        return ensemble_predictions, ensemble_probs

    def update_weights(self, weights: Dict[str, float]):
        """Update ensemble weights"""
        self.weights = weights


class TransformerEnsemble:
    """Ensemble combining BERT with LSTM/GRU models"""

    def __init__(
        self,
        bert_model: BertClassifier,
        lstm_model: Model,
        gru_model: Model,
        weights: Dict[str, float] = None,
    ):
        """Initialize transformer ensemble"""
        self.bert = bert_model
        self.lstm = lstm_model
        self.gru = gru_model
        self.weights = weights or {
            "bert": 0.5,
            "lstm": 0.3,
            "gru": 0.2,
        }

    def predict(self, texts: List[str], sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions

        Args:
            texts: List of texts for BERT
            sequences: Sequence arrays for LSTM/GRU

        Returns:
            Tuple of (predictions, probabilities)
        """
        # BERT predictions
        bert_preds, bert_probs = self.bert.predict(texts)

        # LSTM predictions
        lstm_output = self.lstm.predict(sequences)
        lstm_probs = lstm_output

        # GRU predictions
        gru_output = self.gru.predict(sequences)
        gru_probs = gru_output

        # Ensemble voting
        ensemble_prob = (
            self.weights["bert"] * bert_probs
            + self.weights["lstm"] * lstm_probs
            + self.weights["gru"] * gru_probs
        )

        ensemble_pred = np.argmax(ensemble_prob, axis=1)

        return ensemble_pred, ensemble_prob


def get_transformer_model(
    model_name: str,
    model_dir: str = "results/models",
    load_pretrained: bool = False,
) -> Any:
    """
    Get transformer model

    Args:
        model_name: Name of model (bert, distilbert, roberta)
        model_dir: Directory containing models
        load_pretrained: Whether to load pretrained weights

    Returns:
        Model object
    """
    if load_pretrained:
        model_path = Path(model_dir) / f"{model_name}_model"
        if model_path.exists():
            if TRANSFORMERS_AVAILABLE:
                return TFAutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        if TRANSFORMERS_AVAILABLE:
            return TFAutoModelForSequenceClassification.from_pretrained(model_name)

    raise FileNotFoundError(f"Model not found: {model_name}")
