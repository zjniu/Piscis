Installation
------------

**Basic Installation**

Install ``piscis`` from from `PyPI <https://pypi.org/project/piscis/>`_ with ``pip``.

::

   pip install piscis

By default, this will install the CPU version of JAX. If you would like to run or train ``piscis`` on a GPU or TPU, follow the instructions below.

**Platform-specific Installation**

- CPU-only (Linux/macOS/Windows)

::

   pip install -U piscis jax

- GPU (NVIDIA, CUDA 12)

::

   pip install -U piscis "jax[cuda12]"

- TPU (Google Cloud TPU VM, Google Colab)

::

   pip install -U piscis "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

**Platform Compatibility**

+------------------+--------------------+--------------------+--------------------+-------------------+-----------------------------+
|                  | Linux, x86_64      | Linux, aarch64     | Mac, x86_64        | Mac, aarch64      | Windows, x86_64             |
+==================+====================+====================+====================+===================+=============================+
| **CPU**          | Supported          | Supported          | Supported          | Supported         | Supported                   |
+------------------+--------------------+--------------------+--------------------+-------------------+-----------------------------+
| **NVIDIA GPU**   | Supported          | Supported          | Unsupported        | n/a               | Experimental                |
+------------------+--------------------+--------------------+--------------------+-------------------+-----------------------------+
| **AMD GPU**      | Experimental       | Unsupported        | Experimental       | n/a               | Unsupported                 |
+------------------+--------------------+--------------------+--------------------+-------------------+-----------------------------+
| **Apple GPU**    | n/a                | Unsupported        | n/a                | Experimental      | n/a                         |
+------------------+--------------------+--------------------+--------------------+-------------------+-----------------------------+
| **Intel GPU**    | Experimental       | n/a                | n/a                | n/a               | Unsupported                 |
+------------------+--------------------+--------------------+--------------------+-------------------+-----------------------------+

For more information on platform-specific support and installation, please see `JAX's instructions <https://github.com/google/jax#instructions>`_.

Dependencies
------------

Piscis relies on the following packages, which are automatically installed with ``pip``:

- `deeptile <https://github.com/arjunrajlaboratory/DeepTile>`_
- `flax <https://github.com/google/flax>`_
- `huggingface-hub <https://github.com/huggingface/huggingface_hub>`_
- `imageio <https://github.com/imageio/imageio>`_
- `numba <https://numba.pydata.org/>`_
- `numpy <https://numpy.org/>`_
- `opencv-python <https://opencv.org/>`_
- `pandas <https://pandas.pydata.org/>`_
- `scikit-image <https://scikit-image.org/>`_
- `scipy <https://scipy.org/>`_
- `tqdm <https://github.com/tqdm/tqdm>`_
- `xarray <https://xarray.dev/>`_

Contribution and Support
------------------------

Users can submit questions, report bugs, and contribute to Piscis by creating issues and pull requests via GitHub:  
https://github.com/zjniu/Piscis.
