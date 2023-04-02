import jax.numpy as jnp
import json
import numpy as np
import optax
import orbax.checkpoint

from abc import ABC
from flax import serialization
from flax.core import frozen_dict
from flax.training import orbax_utils, train_state
from functools import partial
from jax import jit, random, value_and_grad
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any

from piscis.data import load_datasets, transform_batch, transform_dataset
from piscis.losses import spots_loss
from piscis.models.spots import SpotsModel


class TrainState(train_state.TrainState, ABC):

    batch_stats: Any
    rng: Any
    epoch: Any


def create_train_state(rng, input_size, tx, variables=None):

    rng, subrng = random.split(rng, 2)
    model = SpotsModel()

    if variables is None:
        variables = model.init({'params': subrng}, np.ones((1, *input_size, 1)))
    else:
        variables = frozen_dict.freeze(variables)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats'],
        rng=rng,
        epoch=jnp.array(-1, dtype=int),
    )


def compute_metrics(deltas_pred, labels_pred, batch, loss_weights):

    rmse, bce, sf1 = spots_loss(deltas_pred, labels_pred, batch['deltas'], batch['labels'], batch['dilated_labels'])
    loss = loss_weights['rmse'] * rmse + loss_weights['bce'] * bce + loss_weights['sf1'] * sf1
    metrics = {
        'rmse': rmse,
        'bce': bce,
        'sf1': sf1,
        'loss': loss
    }

    return metrics


@partial(jit, static_argnums=4)
def loss_fn(params, batch_stats, batch, loss_weights, train):

    variables = {'params': params, 'batch_stats': batch_stats}
    images = batch['images']

    if train:
        (deltas_pred, labels_pred), mutated_vars = \
            SpotsModel().apply(variables, images, train=train, mutable=['batch_stats'])
    else:
        deltas_pred, labels_pred = SpotsModel().apply(variables, images, train=train)
        mutated_vars = None
    metrics = compute_metrics(deltas_pred, labels_pred, batch, loss_weights)
    loss = metrics['loss']

    return loss, (metrics, mutated_vars)


@jit
def train_step(state, batch, loss_weights):

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (_, (metrics, mutated_vars)), grads = grad_fn(state.params, state.batch_stats, batch, loss_weights, train=True)
    state = state.apply_gradients(grads=grads, batch_stats=mutated_vars['batch_stats'])

    return state, metrics


def train_epoch(state, ds, batch_size, loss_weights, epoch_learning_rate, input_size, coords_max_length):

    print(f'Epoch {state.epoch + 1}:')

    state.opt_state.hyperparams['learning_rate'] = jnp.array(epoch_learning_rate, dtype=float)

    rng, *subrngs = random.split(state.rng, 3)
    train_ds = transform_dataset(ds['train'], input_size, key=subrngs[0])
    valid_ds = transform_dataset(ds['valid'], input_size)

    train_ds_size = len(train_ds['images'])
    valid_ds_size = len(valid_ds['images'])
    n_steps = train_ds_size // batch_size
    pbar = tqdm(total=n_steps)

    perms = random.permutation(subrngs[1], train_ds_size)
    perms = perms[:n_steps * batch_size]  # skip incomplete batch
    perms = perms.reshape((n_steps, batch_size))

    batch_metrics = []
    epoch_metrics = {}
    for perm in perms:

        train_batch = {k: v[perm, ...] for k, v in train_ds.items()}
        train_batch = transform_batch(train_batch, coords_max_length)
        state, metrics = train_step(state, train_batch, loss_weights)
        metrics = {k: float(v) for k, v in metrics.items()}
        batch_metrics.append(metrics)

        # compute mean of metrics across each batch in epoch.
        epoch_metrics = {
            k: np.mean([metrics[k] for metrics in batch_metrics]).astype(float)
            for k in batch_metrics[0]}
        epoch_metrics['n_steps'] = n_steps

        summary = (
            f"(train) loss: {epoch_metrics['loss']:>6.4f}, rmse: {epoch_metrics['rmse']:>6.4f}, "
            f"bce: {epoch_metrics['bce']:>6.4f}, sf1: {epoch_metrics['sf1']:>6.4f}"
        )

        pbar.update(1)
        pbar.set_postfix_str(summary)

    val_batch_metrics = []
    val_epoch_metrics = []
    for i in range(valid_ds_size):

        val_batch = {k: v[i:i + 1, ...] for k, v in valid_ds.items()}
        val_batch = transform_batch(val_batch, coords_max_length)
        _, (val_metrics, _) = loss_fn(state.params, state.batch_stats, val_batch, loss_weights, train=False)
        val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
        val_batch_metrics.append(val_metrics)

        val_epoch_metrics = {
            k: np.mean([metrics[k] for metrics in val_batch_metrics]).astype(float)
            for k in val_batch_metrics[0]}

        summary = (
            f"(valid) loss: {val_epoch_metrics['val_loss']:>6.4f}, rmse: {val_epoch_metrics['val_rmse']:>6.4f}, "
            f"bce: {val_epoch_metrics['val_bce']:>6.4f}, sf1: {val_epoch_metrics['val_sf1']:>6.4f} | "
            f"(train) loss: {epoch_metrics['loss']:>6.4f}, rmse: {epoch_metrics['rmse']:>6.4f}, "
            f"bce: {epoch_metrics['bce']:>6.4f}, sf1: {epoch_metrics['sf1']:>6.4f}"
        )

        pbar.set_postfix_str(summary)

    pbar.close()

    epoch_metrics = epoch_metrics | val_epoch_metrics
    epoch_metrics['learning_rate'] = epoch_learning_rate
    state = state.replace(epoch=state.epoch + 1)
    state = state.replace(rng=rng)

    return state, batch_metrics, epoch_metrics


def train_model(model_path, dataset_path, dataset_adjustment='normalize',
                epochs=200, random_seed=0, batch_size=4, learning_rate=0.001,
                warmup_epochs=10, decay_epochs=100, decay_rate=0.5, decay_transition_epochs=10,
                optimizer=None, loss_weights=None):

    if warmup_epochs + decay_epochs > epochs:
        raise ValueError('warmup_epochs + decay_epochs cannot be greater than epochs.')

    model_path = Path(model_path)
    model_parent_path = model_path.parent
    model_name = model_path.stem
    checkpoint_path = model_parent_path.joinpath(f'{model_name}_ckpts')
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    batch_metrics_log_path = model_parent_path.joinpath(f'{model_name}_batch_metrics_log')
    epoch_metrics_log_path = model_parent_path.joinpath(f'{model_name}_epoch_metrics_log')

    if batch_metrics_log_path.is_file():
        with open(batch_metrics_log_path, 'r') as f_batch_metrics_log:
            batch_metrics_log = json.load(f_batch_metrics_log)
    else:
        batch_metrics_log = []
    if epoch_metrics_log_path.is_file():
        with open(epoch_metrics_log_path, 'r') as f_epoch_metrics_log:
            epoch_metrics_log = json.load(f_epoch_metrics_log)
    else:
        epoch_metrics_log = []

    print('Loading datasets...')
    ds = load_datasets(dataset_path, adjustment=dataset_adjustment)
    train_images_shape = ds['train']['images'].shape
    input_size = train_images_shape[1:3]
    coords_max_length = \
        max([len(coords) for coords in ds['train']['coords']] + [len(coords) for coords in ds['valid']['coords']])

    rng = random.PRNGKey(random_seed)

    warmup = [learning_rate * i / warmup_epochs for i in range(warmup_epochs)]
    constant = [learning_rate] * (epochs - warmup_epochs - decay_epochs)
    decay = [learning_rate * decay_rate ** np.ceil(i / decay_transition_epochs) for i in range(1, decay_epochs + 1)]
    schedule = warmup + constant + decay

    if optimizer is None:
        optimizer = partial(optax.sgd, momentum=0.9, nesterov=True)
    tx = optax.inject_hyperparams(optimizer)(learning_rate=learning_rate)

    if loss_weights is None:
        loss_weights = {
            'rmse': 0.4,
            'bce': 0.2,
            'sf1': 1.0
        }

    mgr_options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2)
    checkpointers = {'state': orbax.checkpoint.PyTreeCheckpointer()}
    ckpt_mgr = orbax.checkpoint.CheckpointManager(
        directory=checkpoint_path,
        checkpointers=checkpointers,
        options=mgr_options
    )

    if (next(checkpoint_path.iterdir(), None) is None) and model_path.is_file():
        print(f'Loading existing model weights from {model_path}...')
        with open(model_path, 'rb') as f_model:
            variables = serialization.from_bytes(target=None, encoded_bytes=f_model.read())
    else:
        variables = None

    print('Creating new TrainState...')
    state = create_train_state(rng, input_size, tx, variables)
    latest_epoch = ckpt_mgr.latest_step()
    if latest_epoch is not None:
        print(f'Loading latest checkpoint from {checkpoint_path}...')
        restore_args = orbax_utils.restore_args_from_target(state, mesh=None)
        state = ckpt_mgr.restore(
            step=latest_epoch,
            items={'state': state},
            restore_kwargs={'state': {'restore_args': restore_args}}
        )['state']

    for epoch_learning_rate in schedule:

        state, batch_metrics, epoch_metrics = \
            train_epoch(state, ds, batch_size, loss_weights, epoch_learning_rate, input_size, coords_max_length)

        batch_metrics_log += batch_metrics
        epoch_metrics_log += [epoch_metrics]

        save_args = orbax_utils.save_args_from_target(state)
        ckpt_mgr.save(
            step=state.epoch,
            items={'state': state},
            save_kwargs={'state': {'save_args': save_args}}
        )

        with open(batch_metrics_log_path, 'w') as f_batch_metrics_log:
            json.dump(batch_metrics_log, f_batch_metrics_log, indent=4)
        with open(epoch_metrics_log_path, 'w') as f_epoch_metrics_log:
            json.dump(epoch_metrics_log, f_epoch_metrics_log, indent=4)

    variables = {'params': state.params, 'batch_stats': state.batch_stats, 'input_size': input_size}
    bytes_model = serialization.to_bytes(variables)

    with open(model_path, 'wb') as f_model:
        f_model.write(bytes_model)
