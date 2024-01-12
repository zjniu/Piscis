import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint

from abc import ABC
from flax import serialization
from flax.core import frozen_dict
from flax.training import orbax_utils, train_state
from functools import partial
from jax import jit, random, value_and_grad
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional, Tuple

from piscis.data import load_datasets, transform_batch, transform_subdataset
from piscis.losses import (dice_loss, masked_l2_loss, smoothf1_loss, weighted_bce_loss, weighted_binary_focal_loss,
                           wrap_loss_fn)
from piscis.models.spots import round_input_size, SpotsModel
from piscis.optimizers import sgdw
from piscis.paths import CHECKPOINTS_DIR, MODELS_DIR


class TrainState(train_state.TrainState, ABC):

    """TrainState class used to store the current state of the model during training.
    Inherits from the train_state.TrainState class provided by Flax and adds additional attributes.

    Attributes
    ----------
    batch_stats : Any
        Batch statistics used for normalization.
    key : Any
        Random Key used for training.
    epoch : Any
        Current epoch number.
    """

    batch_stats: Any
    key: Any
    epoch: Any


def create_train_state(
        key: jax.Array,
        input_size: Tuple[int, int],
        dropout_rate: float,
        tx: optax.GradientTransformation,
        variables: Optional[Dict] = None
) -> TrainState:

    """Create a new TrainState object.

    Parameters
    ----------
    key : jax.Array
        Random key used for initialization and training.
    input_size : Tuple[int, int]
        Size of the input images used for training.
    dropout_rate : float
        Dropout rate at skip connections.
    tx : optax.GradientTransformation
        Optax optimizer used for training.
    variables : Optional[Dict]
        Model variables.

    Returns
    -------
    state : TrainState
        New TrainState object.
    """

    # Split the random key.
    key, subkey = random.split(key, 2)

    # Initialize the model.
    model = SpotsModel(dropout_rate=dropout_rate)

    # Initialize parameters.
    if variables is None:
        variables = model.init(subkey, np.ones((1, *input_size, 1)), train=False)
    else:
        variables = frozen_dict.freeze(variables)

    # Create a TrainState object.
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats'],
        key=key,
        epoch=jnp.array(-1, dtype=int),
    )

    return state


def compute_training_metrics(
        deltas_pred: jax.Array,
        labels_pred: jax.Array,
        batch: Dict[str, jax.Array],
        dilation_iterations: int,
        max_distance: float,
        loss_weights: Dict[str, float]
) -> Dict[str, jax.Array]:

    """Compute training metrics.

    Parameters
    ----------
    deltas_pred : jax.Array
        Predicted displacement vectors.
    labels_pred : jax.Array
        Predicted binary labels.
    batch : Dict[str, jax.Array]
        Dictionary containing the input image and target arrays.
    dilation_iterations : int
        Number of iterations used to dilate ground truth labels.
    max_distance : float
        Maximum distance for matching predicted and ground truth displacement vectors.
    loss_weights : Dict[str, float]
        Weights for terms in the overall loss function.

    Returns
    -------
    metrics : Dict[str, jax.Array]
        Dictionary containing the values of individual loss terms and the overall loss.
    """

    # Create the metrics dictionary.
    metrics = {}

    # Compute loss terms.
    if 'l2' in loss_weights:
        metrics['l2'] = wrap_loss_fn(masked_l2_loss)(deltas_pred, batch['deltas'], batch['labels'][:, :, :, 0])
    if 'bce' in loss_weights:
        metrics['bce'] = wrap_loss_fn(weighted_bce_loss)(labels_pred, batch['labels'])
    if 'dice' in loss_weights:
        metrics['dice'] = wrap_loss_fn(dice_loss)(labels_pred, batch['labels'])
    if 'focal' in loss_weights:
        metrics['focal'] = wrap_loss_fn(weighted_binary_focal_loss)(labels_pred, batch['labels'])
    if 'smoothf1' in loss_weights:
        metrics['smoothf1'] = wrap_loss_fn(smoothf1_loss)(deltas_pred, labels_pred, batch['deltas'], batch['labels'],
                                                          dilation_iterations, max_distance)

    # Compute the overall loss.
    metrics['loss'] = jnp.array(sum([loss_weights[k] * v for k, v in metrics.items()]))

    return metrics


@partial(jit, static_argnums=(4, 5, 7))
def loss_fn(
        params: Any,
        state: TrainState,
        batch: Dict[str, jax.Array],
        key: Optional[jax.Array],
        dilation_iterations: int,
        max_distance: float,
        loss_weights: Dict[str, float],
        train: bool
) -> Tuple[jax.Array, Tuple[Dict[str, jax.Array], Dict]]:

    """Computes the loss and metrics for a given batch.

    Parameters
    ----------
    params : Any
        Model parameters.
    state : TrainState
        Current training state.
    batch : Dict[str, jax.Array]
        Dictionary containing the input images and target arrays.
    key : Optional[jax.Array]
        Random Key used for dropout.
    dilation_iterations : int
        Number of iterations used to dilate ground truth labels.
    max_distance : float
        Maximum distance for matching predicted and ground truth displacement vectors.
    loss_weights : Dict[str, float]
        Weights for terms in the overall loss function.
    train : bool
        Whether the model is being trained.

    Returns
    -------
    loss : jax.Array
        Overall loss value.
    aux : Tuple[Dict[str, jax.Array], Dict]
        Auxiliary containing metrics and mutable state variables.
    """

    variables = {'params': params, 'batch_stats': state.batch_stats}
    images = batch['images']

    # Apply the model to the images, using batch_stats as a mutable variable if training.
    if train:
        (deltas_pred, labels_pred), mutated_vars = \
            state.apply_fn(variables, images, train=train, rngs={'dropout': key}, mutable=['batch_stats'])
    else:
        deltas_pred, labels_pred = state.apply_fn(variables, images, train=train)
        mutated_vars = None

    # Compute the loss and metrics.
    metrics = compute_training_metrics(deltas_pred, labels_pred, batch, dilation_iterations, max_distance, loss_weights)
    loss = metrics['loss']
    aux = (metrics, mutated_vars)

    return loss, aux


@partial(jit, static_argnums=(3, 4))
def train_step(
        state: TrainState,
        batch: Dict[str, jax.Array],
        key: Optional[jax.Array],
        dilation_iterations: int,
        max_distance: float,
        loss_weights: Dict[str, float]
) -> Tuple[TrainState, Dict[str, jax.Array]]:

    """Performs a single training step.

    Parameters
    ----------
    state : TrainState
        Current training state.
    batch : Dict[str, jax.Array]
        Dictionary containing the input images and target arrays.
    key : Optional[jax.Array]
        Random Key used for dropout.
    dilation_iterations : int
        Number of iterations used to dilate ground truth labels.
    max_distance : float
        Maximum distance for matching predicted and ground truth displacement vectors.
    loss_weights : Dict[str, float]
        Weights for terms in the overall loss function.

    Returns
    -------
    state : TrainState
        Updated training state.
    metrics : Dict[str, jax.Array]
        Dictionary containing the values of individual loss terms and the overall loss.
    """

    # Define the gradient function.
    grad_fn = value_and_grad(loss_fn, has_aux=True)

    # Compute gradients and update parameters.
    (_, (metrics, mutated_vars)), grads = grad_fn(state.params, state, batch, key,
                                                  dilation_iterations, max_distance, loss_weights, train=True)
    state = state.apply_gradients(grads=grads, batch_stats=mutated_vars['batch_stats'])

    return state, metrics


def train_epoch(
        state: TrainState,
        dataset: Dict,
        input_size: Tuple[int, int],
        batch_size: int,
        epoch_learning_rate: float,
        dilation_iterations: int,
        max_distance: float,
        loss_weights: Dict[str, float],
        coords_max_length: int
) -> Tuple[TrainState, List[Dict[str, float]], Dict[str, float]]:

    """Train the model for a single epoch.

    Parameters
    ----------
    state : TrainState
        Current train state.
    dataset : Dict
        Dataset dictionary.
    input_size : Tuple[int, int]
        Size of the input images used for training.
    batch_size : int
        Batch size for training.
    epoch_learning_rate : float
        Learning rate for the current epoch.
    dilation_iterations : int
        Number of iterations to dilate ground truth labels to minimize class imbalance misclassifications due to minor
        offsets.
    max_distance : float
        Maximum distance for matching predicted and ground truth displacement vectors.
    loss_weights : Dict[str, float]
        Weights for terms in the overall loss function.
    coords_max_length : int
        Maximum length of the coordinates sequence.

    Returns
    -------
    state : TrainState
        Updated train state.
    batch_metrics : List[Dict[str, float]]
        List of metrics computed for each batch.
    epoch_metrics : Dict[str, float]
        Dictionary of metrics computed for epoch.
    """

    print(f'Epoch {state.epoch + 1}:')

    # Update the learning rate.
    state.opt_state.hyperparams['learning_rate'] = jnp.array(epoch_learning_rate, dtype=float)

    # Split the random key and transform the training set.
    key = random.fold_in(state.key, state.epoch)
    subkeys = random.split(key, 3)
    train_ds = transform_subdataset(dataset['train'], input_size, key=subkeys[0])
    valid_ds = dataset['valid']

    train_ds_size = len(train_ds['images'])
    valid_ds_size = len(valid_ds['images'])
    n_steps = train_ds_size // batch_size

    # Initialize the progress bar.
    pbar = tqdm(total=n_steps)

    # Shuffle the training set.
    perms = random.permutation(subkeys[1], train_ds_size)
    perms = perms[:n_steps * batch_size]
    perms = perms.reshape((n_steps, batch_size))

    batch_metrics = []
    epoch_metrics = {}
    summary = None
    for perm in perms:

        # Extract and transform the current training batch.
        train_batch = {k: v[perm] for k, v in train_ds.items()}
        train_batch = transform_batch(train_batch, dilation_iterations, coords_max_length)

        # Perform a training step and update metrics.
        state, metrics = train_step(state, train_batch, subkeys[2], dilation_iterations, max_distance, loss_weights)
        metrics = {k: float(v) for k, v in metrics.items()}
        batch_metrics.append(metrics)

        # Compute mean training metrics across each batch in epoch.
        epoch_metrics = {
            k: np.mean([metrics[k] for metrics in batch_metrics]).astype(float)
            for k in batch_metrics[0]}
        epoch_metrics['n_steps'] = n_steps

        # Update the progress bar.
        summary = f'''(train) loss: {epoch_metrics['loss']:> 6.4f}, 
                      {', '.join([f'{k}: {epoch_metrics[k]:> 6.4f}' for k in loss_weights])}'''
        pbar.update(1)
        pbar.set_postfix_str(summary)

    val_batch_metrics = []
    val_epoch_metrics = []
    for i in range(valid_ds_size):

        # Extract and transform the current validation batch.
        val_batch = {k: v[i:i + 1] for k, v in valid_ds.items()}
        val_batch = transform_batch(val_batch, dilation_iterations, coords_max_length)

        # Compute and update validation metrics.
        _, (val_metrics, _) = loss_fn(state.params, state, val_batch, None,
                                      dilation_iterations, max_distance, loss_weights, train=False)
        val_metrics = {f'val_{k}': float(v) for k, v in val_metrics.items()}
        val_batch_metrics.append(val_metrics)

        # Compute mean validation metrics.
        val_epoch_metrics = {
            k: np.mean([metrics[k] for metrics in val_batch_metrics]).astype(float)
            for k in val_batch_metrics[0]}

        # Update the progress bar.
        val_summary = f'''(valid) loss: {val_epoch_metrics['val_loss']:> 6.4f}, 
                          {', '.join([f"val_{k}: {val_epoch_metrics[f'val_{k}']:> 6.4f}" for k in loss_weights])} | 
                          {summary}'''
        pbar.set_postfix_str(val_summary)

    pbar.close()

    # Compute mean training and validation metrics.
    epoch_metrics = epoch_metrics | val_epoch_metrics
    epoch_metrics['learning_rate'] = epoch_learning_rate

    # Update the training state.
    state = state.replace(epoch=state.epoch + 1)

    return state, batch_metrics, epoch_metrics


def train_model(
        model_name: str,
        dataset_path: str,
        adjustment: Optional[str] = 'standardize',
        input_size: Tuple[int, int] = (256, 256),
        random_seed: int = 0,
        batch_size: int = 4,
        learning_rate: float = 0.2,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.2,
        epochs: int = 400,
        warmup_fraction: float = 0.05,
        decay_fraction: float = 0.5,
        decay_transitions: int = 10,
        decay_factor: float = 0.5,
        dilation_iterations: int = 1,
        max_distance: float = 3.0,
        loss_weights: Optional[Dict[str, float]] = None
) -> None:

    """Train a SpotsModel.

    Parameters
    ----------
    model_name: str
        Name of a new or existing model.
    dataset_path : str
        Path to the directory containing training and validation datasets.
    adjustment : Optional[str], optional
        Adjustment type applied to images. Supported types are 'normalize' and 'standardize'. Default is 'standardize'.
    input_size : Tuple[int, int], optional
        Size of the input images used for training. Default is (256, 256).
    random_seed : int, optional
        Random seed used for initialization and training. Default is 0.
    batch_size : int, optional
        Batch size for training. Default is 4.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 0.2.
    weight_decay : float, optional
        Strength of the weight decay regularization. Default is 1e-4.
    dropout_rate : float, optional
        Dropout rate at skip connections. Default is 0.2.
    epochs : int, optional
        Number of epochs to train the model for. Default is 400.
    warmup_fraction : float, optional
        Fraction of epochs for learning rate warmup. Default is 0.05.
    decay_fraction : float, optional
        Fraction of epochs for learning rate decay. Default is 0.5.
    decay_transitions : int, optional
        Number of times to decay the learning rate. Default is 10.
    decay_factor : float, optional
        Multiplicative factor of each learning rate decay transition. Default is 0.5.
    dilation_iterations : int, optional
        Number of iterations to dilate ground truth labels to minimize class imbalance and misclassifications due to
        minor offsets. Default is 1.
    max_distance : float, optional
        Maximum distance for matching predicted and ground truth displacement vectors. Default is 3.0.
    loss_weights : Optional[Dict[str, float]], optional
        Weights for terms in the overall loss function. Supported terms are 'l2', 'bce', 'focal', 'dice', and
        'smoothf1'. If None, the loss weights {'l2': 0.25, 'smoothf1': 1.0} will be used. Default is None.

    Raises
    ------
    ValueError
        If warmup_fraction + decay_fraction is greater than 1.
    """

    if warmup_fraction + decay_fraction > 1:
        raise ValueError("warmup_fraction + decay_fraction cannot be greater 1.")

    # Load datasets.
    print('Loading datasets...')
    dataset = load_datasets(dataset_path, adjustment, load_train=True, load_valid=True, load_test=False)
    dataset['valid'] = transform_subdataset(dataset['valid'], input_size)
    coords_max_length = max([len(coords) for coords in dataset['train']['coords']] +
                            [len(coords) for coords in dataset['valid']['coords']])

    # Round the input size.
    input_size = round_input_size(input_size)

    # Create the random key.
    key = random.PRNGKey(random_seed)

    # Create the learning rate schedule.
    warmup_epochs = round(warmup_fraction * epochs)
    decay_epochs = round(decay_fraction * epochs)
    decay_transition_epochs = np.ceil(decay_epochs / decay_transitions).astype(int)
    warmup = [learning_rate * i / warmup_epochs for i in range(warmup_epochs)]
    constant = [learning_rate] * (epochs - warmup_epochs - decay_epochs)
    decay = [learning_rate * decay_factor ** np.ceil(i / decay_transition_epochs) for i in range(1, decay_epochs + 1)]
    learning_rate_schedule = warmup + constant + decay

    # Create the optimizer.
    optimizer = partial(sgdw, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    tx = optax.inject_hyperparams(optimizer)(learning_rate=learning_rate)

    # Default loss weights.
    if loss_weights is None:
        loss_weights = {'l2': 0.25, 'smoothf1': 1.0}

    # Define directories for storing checkpoints and the model.
    checkpoint_path = CHECKPOINTS_DIR / model_name
    model_path = MODELS_DIR / model_name

    # Create a checkpoint manager.
    mgr_options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2)
    checkpointers = {
        'state': orbax.checkpoint.PyTreeCheckpointer(),
        'batch_metrics_log': orbax.checkpoint.Checkpointer(orbax.checkpoint.JsonCheckpointHandler()),
        'epoch_metrics_log': orbax.checkpoint.Checkpointer(orbax.checkpoint.JsonCheckpointHandler())
    }
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    ckpt_mgr = orbax.checkpoint.CheckpointManager(
        directory=checkpoint_path,
        checkpointers=checkpointers,
        options=mgr_options
    )

    # Load existing model weights.
    if (next(checkpoint_path.iterdir(), None) is None) and model_path.is_file():
        print(f'Loading existing model weights from {model_path}...')
        with open(model_path, 'rb') as f_model:
            variables = serialization.from_bytes(target=None, encoded_bytes=f_model.read())['variables']
    else:
        variables = None

    # Create a new training state.
    print('Creating new TrainState...')
    state = create_train_state(key, input_size, dropout_rate, tx, variables)

    # Create lists for storing batch and epoch metrics.
    batch_metrics_log = []
    epoch_metrics_log = []

    # Load the latest checkpoint.
    latest_epoch = ckpt_mgr.latest_step()
    if latest_epoch is not None:
        print(f'Loading latest checkpoint from {checkpoint_path}...')
        restore_args = orbax_utils.restore_args_from_target(state, mesh=None)
        checkpoint = ckpt_mgr.restore(
            step=latest_epoch,
            items={'state': state, 'batch_metrics_log': batch_metrics_log, 'epoch_metrics_log': epoch_metrics_log},
            restore_kwargs={'state': {'restore_args': restore_args}}
        )
        state = checkpoint['state']
        batch_metrics_log = checkpoint['batch_metrics_log']
        epoch_metrics_log = checkpoint['epoch_metrics_log']

    for epoch_learning_rate in learning_rate_schedule:

        # Train the model for a single epoch.
        state, batch_metrics, epoch_metrics = \
            train_epoch(state, dataset, input_size, batch_size, epoch_learning_rate,
                        dilation_iterations, max_distance, loss_weights, coords_max_length)

        # Update batch metrics and epoch metrics logs.
        batch_metrics_log += batch_metrics
        epoch_metrics_log += [epoch_metrics]

        # Save a checkpoint.
        save_args = orbax_utils.save_args_from_target(state)
        ckpt_mgr.save(
            step=state.epoch,
            items={'state': state, 'batch_metrics_log': batch_metrics_log, 'epoch_metrics_log': epoch_metrics_log},
            save_kwargs={'state': {'save_args': save_args}}
        )

    # Save the model.
    model_dict = {
        'variables': {
            'params': state.params,
            'batch_stats': state.batch_stats
        },
        'adjustment': adjustment,
        'input_size': input_size,
        'dilation_iterations': dilation_iterations
    }
    bytes_model = serialization.to_bytes(model_dict)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f_model:
        f_model.write(bytes_model)
