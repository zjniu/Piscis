from piscis.core import Piscis


def _initialize_jax_cache() -> None:

    """Initialize the JAX compilation cache."""

    from jax._src import compilation_cache
    from piscis.paths import CACHE_DIR

    # Set the compilation cache directory.
    compilation_cache.set_cache_dir(CACHE_DIR)


_initialize_jax_cache()
del _initialize_jax_cache
