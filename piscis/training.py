import json
import numpy as np
import optax

from abc import ABC
from flax import serialization
from flax.training import checkpoints, train_state
from functools import partial
from jax import jit, random, value_and_grad
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any

from piscis.models.spots import SpotsModel
from piscis.data import load_datasets, transform_batch, transform_dataset
from piscis.losses import spots_loss


class TrainState(train_state.TrainState, ABC):

    batch_stats: Any
    epoch: int
    rng: Any


def create_train_state(rng, input_size, learning_rate, variables=None):

    rng, subrng = random.split(rng, 2)
    model = SpotsModel()

    if variables is None:
        variables = model.init({'params': subrng}, np.ones((1, *input_size, 1)))
    tx = optax.adabelief(learning_rate)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        epoch=-1,
        rng=rng,
        tx=tx,
    )


def compute_metrics(poly_features, batch, loss_weights):

    deltas_pred, labels_pred = poly_features

    rmse, bcel, smoothf1 = spots_loss(deltas_pred, labels_pred,
                                      batch['deltas'], batch['labels'], batch['dilated_labels'])
    loss = loss_weights['rmse'] * rmse + loss_weights['bcel'] * bcel + loss_weights['smoothf1'] * smoothf1
    metrics = {
        'rmse': rmse,
        'bcel': bcel,
        'smoothf1': smoothf1,
        'loss': loss
    }

    return metrics


@partial(jit, static_argnums=2)
def loss_fn(params, batch_stats, train, batch, loss_weights):

    variables = {'params': params, 'batch_stats': batch_stats}
    images = batch['images']

    if train:
        poly_features, mutated_vars = SpotsModel().apply(variables, images, train=train, mutable=['batch_stats'])
    else:
        poly_features = SpotsModel().apply(variables, images, train=train)
        mutated_vars = None
    metrics = compute_metrics(poly_features, batch, loss_weights)
    loss = metrics['loss']

    return loss, (metrics, mutated_vars)


@partial(jit, static_argnums=3)
def train_step(state, train_batch, loss_weights, learning_rate):

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (_, (metrics, mutated_vars)), grads = grad_fn(state.params, state.batch_stats, True, train_batch, loss_weights)
    state = state.apply_gradients(grads=grads, batch_stats=mutated_vars['batch_stats'])
    lr = learning_rate(state.step)
    metrics['learning_rate'] = lr

    return state, metrics


def train_epoch(state, ds, batch_size, loss_weights, learning_rate, input_size, coords_max_length):

    print(f'Epoch: {state.epoch + 1}')

    rng, *subrngs = random.split(state.rng, 3)
    state = state.replace(rng=rng)
    train_ds = transform_dataset(ds['train'], input_size, key=subrngs[0])
    valid_ds = transform_dataset(ds['valid'], input_size)

    train_ds_size = len(train_ds['images'])
    val_ds_size = len(valid_ds['images'])
    n_steps = train_ds_size // batch_size

    perms = random.permutation(subrngs[1], train_ds_size)
    perms = perms[:n_steps * batch_size]  # skip incomplete batch
    perms = perms.reshape((n_steps, batch_size))

    batch_metrics = []
    epoch_metrics = {}
    pbar = tqdm(perms, total=n_steps)
    for perm in pbar:
        train_batch = {k: v[perm, ...] for k, v in train_ds.items()}
        train_batch = transform_batch(train_batch, coords_max_length)
        state, metrics = train_step(state, train_batch, loss_weights, learning_rate)
        metrics = {k: float(v) for k, v in metrics.items()}
        batch_metrics.append(metrics)

        # compute mean of metrics across each batch in epoch.
        epoch_metrics = {
            k: np.mean([metrics[k] for metrics in batch_metrics]).astype(float)
            for k in batch_metrics[0]}
        epoch_metrics['n_steps'] = n_steps

        summary = (
            f"(train) loss: {epoch_metrics['loss']:>6.4f}, rmse: {epoch_metrics['rmse']:>6.4f}, "
            f"bcel: {epoch_metrics['bcel']:>6.4f}, smoothf1: {epoch_metrics['smoothf1']:>6.4f}"
        )
        pbar.write(summary, end='\r')

    val_batch_metrics = []
    val_epoch_metrics = []
    for i in range(val_ds_size):

        val_batch = {k: v[i:i + 1, ...] for k, v in valid_ds.items()}
        val_batch = transform_batch(val_batch, coords_max_length)
        _, (val_metrics, _) = loss_fn(state.params, state.batch_stats, False, val_batch, loss_weights)
        val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
        val_batch_metrics.append(val_metrics)

        val_epoch_metrics = {
            k: np.mean([metrics[k] for metrics in val_batch_metrics]).astype(float)
            for k in val_batch_metrics[0]}

        summary = (
            f"(train) loss: {epoch_metrics['loss']:>6.4f}, rmse: {epoch_metrics['rmse']:>6.4f}, "
            f"bcel: {epoch_metrics['bcel']:>6.4f}, smoothf1: {epoch_metrics['smoothf1']:>6.4f} | "
            f"(valid) loss: {val_epoch_metrics['val_loss']:>6.4f}, rmse: {val_epoch_metrics['val_rmse']:>6.4f}, "
            f"bcel: {val_epoch_metrics['val_bcel']:>6.4f}, smoothf1: {val_epoch_metrics['val_smoothf1']:>6.4f}"
        )
        pbar.write(summary, end='\r')

    print('\n')

    epoch_metrics = epoch_metrics | val_epoch_metrics
    state = state.replace(epoch=state.epoch + 1)

    return state, batch_metrics, epoch_metrics


def train_model(model_path, dataset_path, dataset_adjustment='normalize',
                n_epochs=100, random_seed=0, batch_size=8, learning_rate=None, loss_weights=None):

    model_path = Path(model_path)
    model_parent_path = model_path.parent
    model_name = model_path.stem
    checkpoint_prefix = f'{model_name}_ckpt'
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

    print('Loading datasets...\n')
    ds = load_datasets(dataset_path, adjustment=dataset_adjustment)
    train_images_shape = ds['train']['images'].shape
    n_train_images = train_images_shape[0]
    input_size = train_images_shape[1:3]
    estimated_steps_per_epoch = n_train_images // batch_size
    coords_max_length = \
        max([len(coords) for coords in ds['train']['coords']] + [len(coords) for coords in ds['valid']['coords']])

    rng = random.PRNGKey(random_seed)

    if learning_rate is None:
        learning_rate = optax.exponential_decay(0.001, estimated_steps_per_epoch, 0.95)

    if loss_weights is None:
        loss_weights = {
            'rmse': 2,
            'bcel': 1,
            'smoothf1': 5
        }

    if next(checkpoint_path.iterdir(), None) is None:
        if model_path.is_file():
            print(f'Loading existing model weights from {model_path}...\n')
            variables = checkpoints.restore_checkpoint(model_path, None)
        else:
            variables = None
        print('Creating new TrainState...\n')
        state = create_train_state(rng, input_size, learning_rate, variables)
        checkpoints.save_checkpoint(
            ckpt_dir=checkpoint_path,
            target=state,
            step=state.epoch,
            prefix=checkpoint_prefix,
        )
    else:
        print(f'Loading latest TrainState from {checkpoint_path}...\n')
        state = create_train_state(rng, input_size, learning_rate)
        state = checkpoints.restore_checkpoint(checkpoint_path, state, prefix=checkpoint_prefix)

    for _ in range(n_epochs):

        state, batch_metrics, epoch_metrics = \
            train_epoch(state, ds, batch_size, loss_weights, learning_rate, input_size, coords_max_length)

        batch_metrics_log += batch_metrics
        epoch_metrics_log += [epoch_metrics]

        checkpoints.save_checkpoint(
            ckpt_dir=checkpoint_path,
            target=state,
            step=state.epoch,
            prefix=checkpoint_prefix,
            keep=2,
            keep_every_n_steps=10,
        )

        with open(batch_metrics_log_path, 'w') as f_batch_metrics_log:
            json.dump(batch_metrics_log, f_batch_metrics_log, indent=4)
        with open(epoch_metrics_log_path, 'w') as f_epoch_metrics_log:
            json.dump(epoch_metrics_log, f_epoch_metrics_log, indent=4)

    variables = {'params': state.params, 'batch_stats': state.batch_stats, 'input_size': input_size}
    bytes_model = serialization.to_bytes(variables)

    with open(model_path, 'wb') as f_model:
        f_model.write(bytes_model)
