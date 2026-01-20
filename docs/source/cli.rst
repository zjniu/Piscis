Command-line Interface (CLI)
----------------------------

Piscis also offers a command-line interface (CLI) for inference and training directly from the terminal. The CLI supports two commands: ``predict`` and ``train``.

Predict Command
---------------

The ``predict`` command is used to detect spots in an image or a folder of images using a trained Piscis model. The results are saved as ``.csv`` files containing the coordinates of predicted spots.

**Usage:**

.. code-block:: bash

    piscis predict [--options] input_path output_path

**Positional Arguments:**
  ``input_path``
    Path to image or folder of images.
  ``output_path``
    Path to save predicted spots as CSV.

**Options:**
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Model name. (default: 20251212)
  --batch-size BATCH_SIZE
                        Batch size for the CNN. (default: 1)
  --stack               Whether the input is a stack of images. (default: False)
  --scale SCALE         Scale factor for rescaling the input. (default: 1.0)
  --threshold THRESHOLD
                        Spot detection threshold. (default: 0.5)
  --min-distance MIN_DISTANCE
                        Minimum distance between spots. (default: 1)

**Example:**

.. code-block:: bash

    piscis predict ./images ./predictions --model-name 20251212

This command processes all images in the ``./images`` folder using the ``20251212`` model and saves results in the ``./predictions`` folder.

Train Command
-------------

The ``train`` command is used to train a new Piscis model or fine-tune an existing one with your custom dataset.

**Usage:**

.. code-block:: bash

    piscis train [--options] model_name dataset_path

**Positional Arguments:**
  ``model_name``
    Model name.
  ``dataset_path``
    Path to a dataset or path to a directory containing multiple datasets.

**Options:**
  -h, --help            show this help message and exit
  --initial-model-name INITIAL_MODEL_NAME
                        Name of an existing model to initialize the weights. (default: None)
  --adjustment ADJUSTMENT
                        Adjustment type applied to images. Supported types are 'normalize' and 'standardize'. (default: standardize)
  --input-size INPUT_SIZE INPUT_SIZE
                        Input size used for training. (default: (256, 256))
  --random-seed RANDOM_SEED
                        Random seed used for initialization and training. (default: 0)
  --batch-size BATCH_SIZE
                        Batch size for training. (default: 4)
  --num-workers NUM_WORKERS
                        Number of workers for data loading. (default: 0)
  --learning-rate LEARNING_RATE
                        Learning rate for the optimizer. (default: 0.1)
  --weight-decay WEIGHT_DECAY
                        Strength of the weight decay regularization. (default: 1e-05)
  --epochs EPOCHS       Number of epochs to train the model for. (default: 500)
  --warmup-fraction WARMUP_FRACTION
                        Fraction of epochs for learning rate warmup. (default: 0.04)
  --decay-fraction DECAY_FRACTION
                        Fraction of epochs for learning rate decay. (default: 0.4)
  --l2-loss-weight L2_LOSS_WEIGHT
                        Weight for the masked L2 loss term in the overall loss function. (default: 0.1)
  --decay-transitions DECAY_TRANSITIONS
                        Number of times to decay the learning rate. (default: 10)
  --decay-factor DECAY_FACTOR
                        Multiplicative factor of each learning rate decay transition. (default: 0.5)
  --dilation-iterations DILATION_ITERATIONS
                        Number of iterations to dilate ground truth labels to minimize class imbalance and misclassifications due to minor offsets. (default: 1)
  --max-distance MAX_DISTANCE
                        Maximum distance for matching predicted and ground truth displacement vectors. (default: 3.0)
  --temperature TEMPERATURE
                        Temperature parameter for softmax. (default: 0.05)
  --epsilon EPSILON     Small constant for numerical stability. (default: 1e-07)
  --checkpoint_every CHECKPOINT_EVERY
                        Number of epochs between saving model checkpoints. (default: 10)
  --device DEVICE       Device for training. (default: cuda)

**Example:**

.. code-block:: bash

    piscis train new_model ./dataset

This command trains a model named ``new_model`` using the dataset in ``./dataset`` and default training parameters.
