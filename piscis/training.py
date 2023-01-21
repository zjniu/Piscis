import jax
import jax.numpy as np
import numpy as onp
import optax

from abc import ABC
from dataclasses import replace
from flax.training import train_state
from functools import partial
from tqdm.auto import tqdm
from typing import Any

from piscis.models.spots import SpotsModel
from piscis.data import transform_batch, transform_dataset
from piscis.losses import spots_loss


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


class TrainState(train_state.TrainState, ABC):
    batch_stats: Any
    rng: Any


def create_train_state(rng, learning_rate, variables=None):
    """Creates initial `TrainState`."""
    rng, *subrngs = jax.random.split(rng, 3)
    model = SpotsModel()
    if variables is None:
        variables = model.init({'params': subrngs[0], 'dropout': subrngs[1]}, np.ones((1, 256, 256, 1)))
    tx = optax.adabelief(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        rng=rng,
        tx=tx,
    )


def loss_fn(params, batch_stats, rng, batch, loss_weights):

    if rng is None:
        poly_features = SpotsModel().apply(
            {'params': params, 'batch_stats': batch_stats}, batch['images'],
            train=False
        )
        mutated_vars = None
    else:
        poly_features, mutated_vars = SpotsModel().apply(
            {'params': params, 'batch_stats': batch_stats}, batch['images'],
            train=True, mutable=['batch_stats'], rngs={'dropout': rng}
        )
    metrics = compute_metrics(poly_features, batch, loss_weights)
    loss = metrics['loss']
    return loss, (metrics, mutated_vars)


@partial(jax.jit, static_argnums=5)
def train_step(state, train_batch, val_batch, rng, loss_weights, learning_rate):

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (train_metrics, mutated_vars)), grads = grad_fn(state.params, state.batch_stats, rng, train_batch, loss_weights)
    _, (val_metrics, _) = loss_fn(state.params, state.batch_stats, None, val_batch, loss_weights)
    state = state.apply_gradients(grads=grads, batch_stats=mutated_vars['batch_stats'])
    metrics = train_metrics | {f'val_{k}': v for k, v in val_metrics.items()}
    lr = learning_rate(state.step)
    metrics['learning_rate'] = lr

    return state, metrics


def train_epoch(epoch, state, train_ds, valid_ds, batch_size, loss_weights, learning_rate):
    """Train for a single epoch."""

    print(f'epoch: {epoch}')

    rng, *subrngs = jax.random.split(state.rng, 5)
    state = replace(state, rng=rng)
    train_ds = transform_dataset(train_ds, subrngs[0])
    valid_ds = transform_dataset(valid_ds, subrngs[1])

    train_ds_size = len(train_ds['images'])
    val_ds_size = len(valid_ds['images'])
    steps_per_epoch = train_ds_size // batch_size
    val_batch_size = val_ds_size // steps_per_epoch

    perms = jax.random.permutation(subrngs[2], train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    val_perms = jax.random.permutation(subrngs[3], val_ds_size)
    val_perms = val_perms[:steps_per_epoch * val_batch_size]  # skip incomplete batch
    val_perms = val_perms.reshape((steps_per_epoch, val_batch_size))

    batch_metrics = []
    for perm, val_perm in tqdm(zip(perms, val_perms), total=steps_per_epoch):
        rng, subrng = jax.random.split(rng)
        train_batch = {k: v[perm, ...] for k, v in train_ds.items()}
        val_batch = {k: v[val_perm, ...] for k, v in valid_ds.items()}
        train_batch = transform_batch(train_batch, 1028)
        val_batch = transform_batch(val_batch, 1028)
        state, metrics = train_step(state, train_batch, val_batch, subrng, loss_weights, learning_rate)
        metrics = {k: float(v) for k, v in metrics.items()}
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    epoch_metrics = {
        k: onp.mean([metrics[k] for metrics in batch_metrics]).astype(float)
        for k in batch_metrics[0]}

    summary = (
        f"(train) loss: {epoch_metrics['loss']:>6.3f}, rmse: {epoch_metrics['rmse']:>6.6f}, "
        f"bcel: {epoch_metrics['bcel']:>6.6f}, smoothf1: {epoch_metrics['smoothf1']:>6.3f}\n"
        f"(val)   loss: {epoch_metrics['val_loss']:>6.3f}, rmse: {epoch_metrics['val_rmse']:>6.6f}, "
        f"bcel: {epoch_metrics['val_bcel']:>6.6f}, smoothf1: {epoch_metrics['val_smoothf1']:>6.3f}\n"
    )
    print(summary)

    return state, batch_metrics, epoch_metrics
