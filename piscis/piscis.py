import dask.array as da
import deeptile
import jax.numpy as jnp
import numpy as np

from deeptile import lift, Output
from deeptile.extensions import stitch
from flax.training import checkpoints
from jax import jit
from jax.lib import xla_bridge
from pathlib import Path
from skimage.transform import resize

from piscis.models.spots import SpotsModel
from piscis import utils

TRAINED_MODELS_DIR = Path(__file__).parent.joinpath('trained_models')


class Piscis:

    def __init__(self, model='spots', batch_size=8):

        if xla_bridge.get_backend().platform == 'cpu':
            batch_size = 1

        self.model_name = model
        self.model = SpotsModel()
        self.variables = checkpoints.restore_checkpoint(TRAINED_MODELS_DIR.joinpath(model), None)
        self.batch_size = batch_size

        @jit
        def jitted(x):

            x = jnp.expand_dims(x, axis=-1)
            deltas, labels = self.model.apply(self.variables, x, False)
            counts = utils.vmap_colocalize_pixels(deltas, labels[:, :, :, 0], (3, 3))

            return deltas, labels, counts

        jitted(jnp.zeros((self.batch_size, 256, 256)))
        self._jitted = jitted

    def predict(self, x, stack=False, scale=1, threshold=2.0, min_distance=1, normalize=True, intermediates=False):

        x, batch_axis, x_min, x_max = _preprocess(x, stack, normalize=normalize)

        dt = deeptile.load(x, link_data=False, dask=True)
        tile_size = (round(256 / scale), ) * 2
        scales = np.array([[255, 255]]) / (np.array(tile_size) - 1)
        tiles = dt.get_tiles(tile_size=tile_size, overlap=(0.1, 0.1)).pad(mode='reflect')

        if x_min is not None:
            tiles = lift(lambda t: (t - x_min) / (x_max - x_min + 1e-7))(tiles)

        if stack and batch_axis:

            batch_axis_len = x.shape[0]
            stack_axis_len = x.shape[-3]
            coords = np.empty(batch_axis_len, dtype=object)
            if intermediates:
                y = []
            else:
                y = None
            for i in range(batch_axis_len):
                output = self._predict_stack(tiles.s[i], stack_axis_len, scales, threshold, min_distance, intermediates)
                if intermediates:
                    c = stitch.stitch_coords(output[0])
                    y.append(np.asarray(stitch.stitch_image(output[1], blend=False)))
                else:
                    c = stitch.stitch_coords(output)
                coords[i] = np.asarray(c)
            if intermediates:
                y = np.stack(y)

        else:

            if stack:
                stack_axis_len = x.shape[-3]
                output = self._predict_stack(tiles, stack_axis_len, scales, threshold, min_distance, intermediates)
            else:
                output = lift(self._process_plane, vectorized=True, batch_axis=batch_axis, pad_final_batch=True,
                             batch_size=self.batch_size)(tiles, scales, threshold, min_distance, intermediates)
            if intermediates:
                coords = stitch.stitch_coords(output[0])
                y = np.asarray(stitch.stitch_image(output[1], blend=False))
            else:
                coords = stitch.stitch_coords(output)
                y = None
            coords = np.asarray(coords)

        if intermediates:
            return coords, y
        else:
            return coords

    def _predict_stack(self, tiles, stack_axis_len, scales, threshold, min_distance, intermediates):

        process_stack = lift(self._process, vectorized=True, batch_axis=True, pad_final_batch=True,
                             batch_size=self.batch_size)
        postprocess_stack = lift(self._postprocess_stack, vectorized=False, batch_axis=False)
        y, process_variables = process_stack.init(tiles, intermediates)
        coords = None
        postprocess_variables = None
        n_steps = process_variables['n_steps'] + 1
        mod = 0
        j = 0
        k = 0

        for i in range(n_steps):

            y, process_variables = process_stack.apply(y, process_variables)

            if postprocess_variables is None:
                coords, postprocess_variables = postprocess_stack.init(y, scales, threshold, min_distance)

            mod = mod + self.batch_size
            if mod >= stack_axis_len:
                coords, postprocess_variables = postprocess_stack.apply(coords, postprocess_variables)
                if not intermediates:
                    y['deltas'][j, k] = None
                    y['counts'][j, k] = None
                mod = mod - stack_axis_len
                if k == coords.shape[1] - 1:
                    k = 0
                    j = j + 1
                else:
                    k = k + 1

        if intermediates:
            y = np.concatenate((y['deltas'], y['labels'], y['counts'].s[:, :, :, None]), axis=-1)
            y = np.moveaxis(y, -1, 1)
            return coords, y
        else:
            return coords

    def _process_plane(self, tiles, scales, threshold, min_distance, intermediates):

        y = self._process(tiles, intermediates)
        deltas = y['deltas']
        counts = y['counts']

        coords = np.empty(len(deltas), dtype=object)
        for i, (d, c) in enumerate(zip(deltas, counts)):
            coords[i] = utils.compute_spot_coordinates(d, c,
                                                       threshold=threshold, min_distance=min_distance)
            coords[i][:, -2:] = coords[i][:, -2:] / scales
        coords = Output(coords, isimage=False, stackable=False, tile_scales=(1.0, 1.0))

        if intermediates:
            y = np.concatenate((y['deltas'], y['labels'], y['counts'][:, :, :, None]), axis=-1)
            y = np.moveaxis(y, -1, 1)
            return coords, y
        else:
            return coords

    def _process(self, tiles, intermediates):

        tiles = tiles.compute()
        if tiles.shape[1:3] != (256, 256):
            tiles = resize(tiles, (tiles.shape[0], 256, 256))

        if intermediates:
            deltas, labels, counts = self._jitted(tiles)
            deltas = np.asarray(deltas)
            labels = np.asarray(labels)
            counts = np.asarray(counts)
            y = {
                'deltas': deltas,
                'labels': labels,
                'counts': counts
            }
        else:
            deltas, _, counts = self._jitted(tiles)
            deltas = np.asarray(deltas)
            counts = np.asarray(counts)
            y = {
                'deltas': deltas,
                'counts': counts
            }

        return y

    @staticmethod
    def _postprocess_stack(y, scales, threshold, min_distance):

        deltas = y['deltas']
        counts = y['counts']

        coords = utils.compute_spot_coordinates(deltas, counts, threshold=threshold, min_distance=min_distance)
        coords[:, -2:] = coords[:, -2:] / scales
        coords = Output(coords, isimage=False, stackable=False, tile_scales=(1.0, 1.0))

        return coords


def _preprocess(x, stack, normalize):

    x = da.from_array(x)

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

    if normalize:
        axis = tuple(range(ndim - nnormdim, ndim))
        stat_shape = (*x.shape[:-nnormdim], *((1, ) * nnormdim))
        x_min = np.min(x, axis=axis).reshape(stat_shape).compute()
        x_max = np.max(x, axis=axis).reshape(stat_shape).compute()
    else:
        x_min = None
        x_max = None

    return x, batch_axis, x_min, x_max
