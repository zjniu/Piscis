Training
--------

This guide on training Piscis follows the code from the `train_piscis.ipynb <https://github.com/zjniu/Piscis/blob/main/notebooks/train_piscis.ipynb>`_ notebook.

**Step 1: Import Required Libraries**

First, import the necessary Piscis modules for training.

.. code:: ipython3

    from piscis.downloads import download_dataset
    from piscis.training import train_model

**Step 2: Download the Piscis Dataset**

Download the dataset required for this example. Here, we use the dataset labeled ``20230905``, which is the dataset used to train and test the model from our paper. The :any:`download_dataset <downloads.download_dataset>` function downloads the specific dataset from our `Hugging Face Dataset Repository <https://huggingface.co/datasets/wniu/Piscis>`_.

.. code:: ipython3

    download_dataset('20230905', '')

You will notice progress bars as each ``.npz`` file is downloaded. These ``.npz`` files contain fluorescence microscopy images with spots and their corresponding ground truth annotations.

**Step 3: Train Piscis**

Train a new Piscis model using the :any:`train_model <training.train_model>` function. The parameters shown below are exactly the same as what we used to train the ``20230905`` model.

.. code:: ipython3

    train_model(
        model_name='new_model',
        dataset_path='20230905',
        adjustment='standardize',
        input_size=(256, 256),
        random_seed=0,
        batch_size=4,
        learning_rate=0.2,
        weight_decay=1e-4,
        dropout_rate=0.2,
        epochs=400,
        warmup_fraction=0.05,
        decay_fraction=0.5,
        decay_transitions=10,
        decay_factor=0.5,
        dilation_iterations=1,
        max_distance=3.0,
        loss_weights={'l2': 0.25, 'smoothf1': 1.0}
    )

See the API reference for the :any:`train_model <training.train_model>` function for more information on each training parameter.

Once training is complete, ``new_model`` will be saved to the ``.piscis/models`` folder in the user's home directory, which is then accessible by the :any:`Piscis <core.Piscis>` class for inference.

Custom Datasets
---------------

In addition to our preformatted datasets, you can create your own custom datasets using the :any:`generate_dataset <data.generate_dataset>` function.

For users of NimbusImage who would like to convert exported annotations into a custom dataset, see the `generate_dataset.ipynb <https://github.com/zjniu/Piscis/blob/main/paper/0_generate_datasets/generate_dataset.ipynb>`_ notebook for an example.

Fine-tuning
-----------

Instead of training a new model from scratch, you may want to consider fine-tuning a pre-trained model such as ``20230905``.

Piscis allows you to initialize training with the weights of an existing model via the ``initial_model_name`` parameter of the :any:`train_model <training.train_model>` function.
