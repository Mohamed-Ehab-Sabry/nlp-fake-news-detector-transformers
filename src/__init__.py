"""
NLP Fake News Detector - Source Module

Organized by functionality:
- data: Data loading and preprocessing
- features: Feature engineering
- models: ML/DL models (classical, deep learning, transformers)
"""

from . import data
from . import features
from . import models

__version__ = "1.0.0"
__author__ = "NLP Team"

__all__ = ["data", "features", "models"]
