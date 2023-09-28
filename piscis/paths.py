from jax._src import compilation_cache
from pathlib import Path

# Define Piscis paths.
PISCIS_DIR = Path.home() / '.piscis'
CACHE_DIR = PISCIS_DIR / 'cache'
CHECKPOINTS_DIR = PISCIS_DIR / 'checkpoints'
MODELS_DIR = PISCIS_DIR / 'models'

# Define Hugging Face paths.
HF_DATASETS_DIR = 'datasets/wniu/Piscis/'
HF_MODELS_DIR = 'wniu/Piscis/models/'


def initialize_cache() -> None:

    """Initialize compilation cache for JAX."""

    compilation_cache.initialize_cache(CACHE_DIR)
