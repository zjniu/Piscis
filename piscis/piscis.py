import deeptile
import jax.numpy as np
import numpy as onp

from deeptile import lift, Output
from deeptile.extensions import stitch
from flax.training import checkpoints
from functools import partial
from jax import jit
from pathlib import Path
from skimage.transform import resize

TRAINED_MODELS_DIR = Path(__file__).parent.joinpath('trained_models')


class Piscis:

    def __init__(self, model='spots', batch_size=8):

        from piscis.models.spots import SpotsModel
        from piscis import utils
        from jax.lib import xla_bridge

        if xla_bridge.get_backend().platform == 'cpu':
            batch_size = 1

        self.model_name = model
        self.model = SpotsModel()
        self.variables = checkpoints.restore_checkpoint(TRAINED_MODELS_DIR.joinpath(model), None)
        self.batch_size = batch_size

        @jit
        def jitted(x):

            x = np.expand_dims(x, axis=-1)
            deltas, labels = self.model.apply(self.variables, x, False)
            counts = utils.vmap_colocalize_pixels(deltas, labels[:, :, :, 0])
            y = np.concatenate((deltas, labels, counts[:, :, :, None]), axis=-1)
            y = np.moveaxis(y, -1, 1)

            return y

        def process(tiles):

            y = jitted(tiles)
            y = onp.asarray(y)

            return y

        process(np.zeros((self.batch_size, 256, 256)))
        self.process = process

        def postprocess(tile, threshold, min_distance):

            deltas = onp.moveaxis(tile[..., :2, :, :], -3, -1)
            counts = tile[..., 3, :, :]
            coords = utils.compute_spot_coordinates(deltas, counts, threshold=threshold, min_distance=min_distance)
            coords = Output(coords, isimage=False, stackable=False)

            return coords

        self.postprocess = postprocess

    def predict(self, x, stack=False, scale=1, threshold=2.0, min_distance=1):

        y, shape, batch_axis = self._predict(x, stack, scale)

        dt = deeptile.load(y, link_data=False, dask=False)
        tiles = dt.get_tiles(tile_size=(256, 256), overlap=(0.1, 0.1))
        coords = lift(partial(self.postprocess,
                              min_distance=min_distance, threshold=threshold), batch_axis=batch_axis)(tiles)
        coords = stitch.stitch_coords(coords)
        coords = onp.asarray(coords)

        if scale != 1:
            scales = (onp.array(y.shape[-2:]) - 1) / (onp.array(shape[-2:]) - 1)
            for i in range(len(coords)):
                coords[i][-2:] = (coords[i][-2:]) / scales

        return coords, y

    def _predict(self, x, stack, scale):

        x, shape, batch_axis = _preprocess(x, stack, scale, normalize=True)

        dt = deeptile.load(x, link_data=False, dask=False)
        tiles = dt.get_tiles(tile_size=(256, 256), overlap=(0.1, 0.1)).pad(mode='reflect')

        if stack:
            tiles = onp.reshape(tiles, (-1, 256, 256))

        tiles = lift(self.process, vectorized=True, batch_axis=batch_axis or stack, pad_final_batch=True,
                     batch_size=self.batch_size)(tiles)
        y = stitch.stitch_image(tiles)

        if stack:
            y = y.reshape(*shape[:-2], *y.shape[-3:])

        return y, shape, batch_axis


class LoG:

    def __init__(self):

        from piscis.models import baseline

        def process(tile, sigma):

            y = baseline.log_filter(tile, sigma)

            return y

        self.process = process

        def postprocess(tile, threshold, min_distance):

            coords = baseline.compute_spot_coordinates(tile, threshold=threshold, min_distance=min_distance)
            coords = Output(coords, isimage=False, stackable=False)

            return coords

        self.postprocess = postprocess

    def predict(self, x, stack=False, scale=1, sigma=1, threshold=0.05, min_distance=1):

        y, shape, batch_axis = self._predict(x, stack, scale, sigma)

        dt = deeptile.load(y, link_data=False, dask=False)
        tiles = dt.get_tiles(tile_size=(256, 256), overlap=(0.1, 0.1)).pad(mode='reflect')
        coords = lift(partial(self.postprocess,
                              threshold=threshold, min_distance=min_distance), batch_axis=batch_axis)(tiles)
        coords = stitch.stitch_coords(coords)

        if scale != 1:
            scales = (onp.array(y.shape[-2:]) - 1) / (onp.array(shape[-2:]) - 1)
            for i in range(len(coords)):
                coords[i][-2:] = (coords[i][-2:]) / scales

        return coords, y

    def _predict(self, x, stack, scale, sigma):

        x, shape, batch_axis = _preprocess(x, stack, scale, normalize=True)

        dt = deeptile.load(x, link_data=False)
        tiles = dt.get_tiles(tile_size=(256, 256), overlap=(0.1, 0.1))

        if stack:
            tiles = lift(lambda tile: tile.reshape(-1, *tile.shape[-2:]))(tiles)

        tiles = lift(partial(self.process, sigma=sigma), batch_axis=batch_axis)(tiles)
        y = stitch.stitch_image(tiles)

        if stack:
            y = y.reshape(*shape[:-2], *y.shape[-2:])

        return y, shape, batch_axis


def _preprocess(x, stack, scale, normalize):

    shape = x.shape
    ndim = x.ndim

    if stack:
        nnormdim = 3
    else:
        nnormdim = 2

    if ndim == nnormdim:
        batch_axis = False
    elif ndim == nnormdim + 1:
        batch_axis = True
    else:
        raise ValueError("Input does not have the correct dimensions.")

    if scale != 1:
        x = resize(x, (*shape[:-2], round(shape[-2] * scale), round(shape[-1] * scale)))

    if normalize:
        axis = tuple(range(ndim - nnormdim, ndim))
        stat_shape = (*x.shape[:-nnormdim], *((1, ) * nnormdim))
        x = (x - onp.min(x, axis=axis).reshape(stat_shape)) / (onp.ptp(x, axis=axis).reshape(stat_shape) + 1e-7)

    return x, shape, batch_axis
