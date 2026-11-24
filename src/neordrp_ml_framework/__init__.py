"""NeoRdRp-ML-Framework: composable models for RdRp biology.

The package exposes four layers:

* ``nn``      -- interchangeable encoders, heads, and composed task models
* ``data``    -- tokenizer, record readers, datasets, and collators
* ``train``   -- generic trainer and task losses
* ``eval``    -- fast biological performance reports

Everything is resolved through a name-based registry, so a YAML file can
replace any component without changing Python code.
"""

from __future__ import annotations

from .nn.models import Detector, FamilyClassifier, Localizer, build_model

__version__ = "0.1.0"

__all__ = [
    "Detector",
    "FamilyClassifier",
    "Localizer",
    "build_model",
    "__version__",
]
