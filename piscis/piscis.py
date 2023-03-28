import dask.array as da
import deeptile
import jax.numpy as jnp
import numpy as np

from deeptile import lift, Output
from deeptile.extensions import stitch
from flax import serialization
from jax import jit
from jax.lib import xla_bridge
from pathlib import Path
from skimage.transform import resize

from piscis.models.spots import SpotsModel
from piscis import utils

TRAINED_MODELS_DIR = Path(__file__).parent.joinpath('trained_models')


class Piscis:

    def __init__(self, model='spots', batch_size=4, input_size=None):

        if xla_bridge.get_backend().platform == 'cpu':
            batch_size = 1

        self.model_name = model
        self.model = SpotsModel()

        with open(TRAINED_MODELS_DIR.joinpath(model), 'rb') as f_model:
            self.variables = serialization.from_bytes(target=None, encoded_bytes=f_model.read())

        self.batch_size = batch_size

        if input_size is None:
            input_size = self.variables['input_size']
            input_size = (input_size['0'], input_size['1'])
        self.input_size = input_size

        @jit
        def jitted(x):

            x = jnp.expand_dims(x, axis=-1)
            deltas, labels = self.model.apply(self.variables, x, False)

            return deltas, labels

        jitted(jnp.zeros((self.batch_size, *self.input_size)))
        self._jitted = jitted

    def predict(self, x, stack=False, scale=1, threshold=2.0, min_distance=1, normalize=True, intermediates=False):

        x, batch_axis, x_min, x_max = _preprocess(x, stack, normalize=normalize)

        dt = deeptile.load(x, link_data=False, dask=True)
        tile_size = (round(self.input_size[0] / scale), round(self.input_size[1] / scale))
        scales = (np.array([self.input_size]) - 1) / (np.array(tile_size) - 1)
        tiles = dt.get_tiles(tile_size=tile_size, overlap=(0.1, 0.1)).pad(mode='symmetric')

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
        carry, process_variables = process_stack.init(tiles)
        coords = None
        postprocess_variables = None
        n_steps = process_variables['n_steps'] + 1
        mod = 0
        j = 0
        k = 0

        for i in range(n_steps):

            carry, process_variables = process_stack.apply(carry, process_variables)

            if postprocess_variables is None:
                coords, postprocess_variables = postprocess_stack.init(*carry, scales, threshold, min_distance)

            mod = mod + self.batch_size
            if mod >= stack_axis_len:
                coords, postprocess_variables = postprocess_stack.apply(coords, postprocess_variables)
                if not intermediates:
                    carry[0][j, k] = None
                    carry[1][j, k] = None
                mod = mod - stack_axis_len
                if k == coords.shape[1] - 1:
                    k = 0
                    j = j + 1
                else:
                    k = k + 1

        if intermediates:
            y = np.concatenate((carry[0], carry[1]), axis=-1)
            y = np.moveaxis(y, -1, 1)
            return coords, y
        else:
            return coords

    def _process_plane(self, tiles, scales, threshold, min_distance, intermediates):

        deltas, labels = self._process(tiles)

        coords = np.empty(len(deltas), dtype=object)
        for i, (d, l) in enumerate(zip(deltas, labels)):
            coords[i] = utils.compute_spot_coordinates(d, l[:, :, 0], threshold=threshold, min_distance=min_distance)
            coords[i][:, -2:] = coords[i][:, -2:] / scales
        coords = Output(coords, isimage=False, stackable=False, tile_scales=(1.0, 1.0))

        if intermediates:
            y = np.concatenate((deltas, labels), axis=-1)
            y = np.moveaxis(y, -1, 1)
            return coords, y
        else:
            return coords

    def _process(self, tiles):

        tiles = tiles.compute()
        if tiles.shape[1:3] != self.input_size:
            tiles = resize(tiles, (tiles.shape[0], *self.input_size))

        deltas, labels = self._jitted(tiles)
        deltas = np.asarray(deltas)
        labels = np.asarray(labels)

        return deltas, labels

    @staticmethod
    def _postprocess_stack(deltas, labels, scales, threshold, min_distance):

        coords = utils.compute_spot_coordinates(deltas, labels[:, :, :, 0],
                                                threshold=threshold, min_distance=min_distance)
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
