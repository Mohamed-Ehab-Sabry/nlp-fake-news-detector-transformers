"""
NLP Fake News Detection - Model Modules
Models organized by training phase and approach
"""

from .classical_models import ClassicalModels, get_classical_model
from .deep_learning import CNN, Autoencoder, get_dl_model
from .transfer_learning import TransferLearningModel, get_pretrained_embedding
from .transformer_models import BertClassifier, EnsembleModel, get_transformer_model

__version__ = "1.0.0"

__all__ = [
    "ClassicalModels",
    "CNN",
    "Autoencoder",
    "TransferLearningModel",
    "BertClassifier",
    "EnsembleModel",
    "get_classical_model",
    "get_dl_model",
    "get_pretrained_embedding",
    "get_transformer_model",
]
