import dask.array as da
import deeptile
import jax.numpy as jnp
import numpy as np

from deeptile import lift, Output
from deeptile.core.data import Tiled
from deeptile.extensions import stitch
from flax import serialization
from jax import jit
from jax.lib import xla_bridge
from pathlib import Path
from skimage.transform import resize
from typing import Dict, Optional, Sequence, Tuple, Union

from piscis.models.spots import SpotsModel
from piscis import utils

TRAINED_MODELS_DIR = Path(__file__).parent.joinpath('trained_models')


class Piscis:

    """Class for running the Piscis algorithm.

    Attributes
    ----------
    model_name : str
        Model name.
    model : SpotsModel
        Instance of the SpotsModel class.
    variables : Dict
        Model variables.
    batch_size : int
        Batch size for the CNN.
    input_size : Sequence[int]
        Input size for the CNN.
    _jitted : Callable
        Compiled model apply function.
    """

    def __init__(
            self,
            model: str = 'spots',
            batch_size: int = 4,
            input_size: Optional[Sequence[int]] = None
    ) -> None:

        """Initialize a Piscis object and compile the model.

        Parameters
        ----------
        model : str, optional
            Model name. Default is 'spots'.
        batch_size : int, optional
            Batch size for the CNN. Default is 4.
        input_size : Optional[Sequence[int]], optional
            Input size for the CNN. If None, it is obtained from the model's variables. Default is None.
        """

        # Set the batch size to 1 if running on CPU.
        if xla_bridge.get_backend().platform == 'cpu':
            batch_size = 1

        # Load the model.
        self.model_name = model
        self.model = SpotsModel()
        with open(TRAINED_MODELS_DIR.joinpath(model), 'rb') as f_model:
            self.variables = serialization.from_bytes(target=None, encoded_bytes=f_model.read())

        # Set the batch size and input size.
        self.batch_size = batch_size
        if input_size is None:
            input_size = self.variables['input_size']
            input_size = (input_size['0'], input_size['1'])
        self.input_size = input_size

        # Define the jitted model apply function.
        @jit
        def jitted(x):

            x = jnp.expand_dims(x, axis=-1)
            deltas, labels = self.model.apply(self.variables, x, False)

            return deltas, labels

        # Compile the model apply function.
        jitted(jnp.zeros((self.batch_size, *self.input_size)))
        self._jitted = jitted

    def predict(
            self,
            x: Union[np.ndarray, da.Array],
            stack: bool = False,
            scale: float = 1.0,
            threshold: float = 0.5,
            min_distance: int = 1,
            normalize: bool = True,
            intermediates: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:

        """Predict spots in an image or stack of images.

        Parameters
        ----------
        x : np.ndarray or da.Array
            Image or stack of images.
        stack : bool, optional
            Whether `x` is a stack of images. Default is False.
        scale : float, optional
            Scale factor for rescaling `x`. Default is 1.
        threshold : float, optional
            Detection threshold between 0 and 1. Default is 0.5.
        min_distance : int, optional
            Minimum distance between spots. Default is 1.
        normalize : bool, optional
            Whether to normalize `x`. Default is True.
        intermediates : bool, optional
            Whether to return intermediate feature maps. Default is False.

        Returns
        -------
        coords : np.ndarray
            Coordinates of detected spots.
        y : np.ndarray
            Intermediate feature maps. Only returned if `intermediates` is True.
        """

        # Preprocess the input.
        x, batch_axis, x_min, x_max = _preprocess(x, stack, normalize=normalize)

        # Create a DeepTile object and get tiles.
        dt = deeptile.load(x, link_data=False, dask=True)
        tile_size = (round(self.input_size[0] / scale), round(self.input_size[1] / scale))
        scales = (np.array([self.input_size]) - 1) / (np.array(tile_size) - 1)
        tiles = dt.get_tiles(tile_size=tile_size, overlap=(0.1, 0.1)).pad(mode='symmetric')

        # Normalize tiles if necessary.
        if x_min is not None:
            tiles = lift(lambda t: (t - x_min) / (x_max - x_min + 1e-7))(tiles)

        # Predict spots in tiles and stitch.
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
                output = lift(self._predict_plane, vectorized=True, batch_axis=batch_axis, pad_final_batch=True,
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

    def _predict_stack(
            self,
            tiles: Tiled,
            stack_axis_len: int,
            scales: np.ndarray,
            threshold: float,
            min_distance: int,
            intermediates: bool
    ) -> Union[Tuple[Output, np.ndarray], Output]:

        """Predict spots in a stack of images.

        Parameters
        ----------
        tiles : Tiled
            Tiles of images.
        stack_axis_len : int
            Length of the stack axis.
        scales : np.ndarray
            Scales for rescaling tiles.
        threshold : float
            Detection threshold between 0 and 1.
        min_distance : int
            Minimum distance between spots.
        intermediates : bool
            Whether to return intermediate feature maps.

        Returns
        -------
        coords : Output
            Coordinates of detected spots.
        y : np.ndarray
            Intermediate feature maps. Only returned if `intermediates` is True.
        """

        # Lift process and postprocess functions.
        process_stack = lift(self._process, vectorized=True, batch_axis=True, pad_final_batch=True,
                             batch_size=self.batch_size)
        postprocess_stack = lift(self._postprocess_stack, vectorized=False, batch_axis=False)

        # Initialize the lifted process function.
        carry, process_variables = process_stack.init(tiles)
        coords = None
        postprocess_variables = None
        i_max = tiles.shape[0]
        mod = 0
        i = 0
        j = 0

        for _ in range(process_variables['n_steps']):

            # Apply the lifted process function.
            carry, process_variables = process_stack.apply(carry, process_variables)

            # Initialize the lifted postprocess function if necessary.
            if postprocess_variables is None:
                coords, postprocess_variables = postprocess_stack.init(*carry, scales, threshold, min_distance)

            # Apply the lifted postprocess function.
            mod = mod + self.batch_size
            while (mod >= stack_axis_len) and (i < i_max):
                coords, postprocess_variables = postprocess_stack.apply(coords, postprocess_variables)
                if not intermediates:
                    carry[0][i, j] = None
                    carry[1][i, j] = None
                mod = mod - stack_axis_len
                if j == coords.shape[1] - 1:
                    i = i + 1
                    j = 0
                else:
                    j = j + 1

        if intermediates:
            y = np.concatenate((carry[0], carry[1]), axis=-1)
            y = np.moveaxis(y, -1, 1)
            return coords, y
        else:
            return coords

    def _predict_plane(
            self,
            tiles: Tiled,
            scales: np.ndarray,
            threshold: float,
            min_distance: int,
            intermediates: bool
    ) -> Union[Tuple[Output, np.ndarray], Output]:

        """Predict spots in a single plane.

        Parameters
        ----------
        tiles : Tiled
            Tiles of images.
        scales : np.ndarray
            Scales for rescaling tiles.
        threshold : float
            Detection threshold between 0 and 1.
        min_distance : int
            Minimum distance between spots.
        intermediates : bool
            Whether to return intermediate feature maps.

        Returns
        -------
        coords : Output
            Coordinates of detected spots.
        y : np.ndarray
            Intermediate feature maps. Only returned if `intermediates` is True.
        """

        # Process tiles.
        deltas, labels = self._process(tiles)

        # Postprocess tiles.
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

    def _process(
            self,
            tiles: Tiled
    ) -> Tuple[np.ndarray, np.ndarray]:

        """Process tiles.

        Parameters
        ----------
        tiles : Tiled
            Tiles of images.

        Returns
        -------
        deltas : np.ndarray
            Predicted subpixel displacements.
        labels : np.ndarray
            Predicted binary labels.
        """

        # Compute Dask arrays if necessary.
        tiles = tiles.compute()

        # Resize tiles if necessary.
        if tiles.shape[1:3] != self.input_size:
            tiles = resize(tiles, (tiles.shape[0], *self.input_size))

        # Run the jitted model apply function on tiles.
        deltas, labels = self._jitted(tiles)
        deltas = np.asarray(deltas)
        labels = np.asarray(labels)

        return deltas, labels

    @staticmethod
    def _postprocess_stack(
            deltas: np.ndarray,
            labels: np.ndarray,
            scales: np.ndarray,
            threshold: float,
            min_distance: int
    ) -> Output:

        """Postprocess tiles.

        Parameters
        ----------
        deltas : np.ndarray
            Predicted subpixel displacements.
        labels : np.ndarray
            Predicted binary labels.
        scales : np.ndarray
            Scales for rescaling tiles.
        threshold : float
            Detection threshold between 0 and 1.
        min_distance : int
            Minimum distance between spots.

        Returns
        -------
        coords : Output
            Coordinates of detected spots.
        """

        # Compute coordinates of detected spots.
        coords = utils.compute_spot_coordinates(deltas, labels[:, :, :, 0],
                                                threshold=threshold, min_distance=min_distance)

        # Rescale coordinates.
        coords[:, -2:] = coords[:, -2:] / scales
        coords = Output(coords, isimage=False, stackable=False, tile_scales=(1.0, 1.0))

        return coords


def _preprocess(
        x: Union[np.ndarray, da.Array],
        stack: bool,
        normalize: bool,
) -> Tuple[da.Array, bool, Optional[np.ndarray], Optional[np.ndarray]]:

    """Preprocess the input.

    Parameters
    ----------
    x : np.ndarray or dask.array.Array
        Image or stack of images.
    stack : bool
        Whether `x` is a stack of images.
    normalize : bool
        Whether to normalize `x`.

    Returns
    -------
    x : dask.array.Array
        Preprocessed image or stack of images.
    batch_axis : bool
        Whether `x` has a batch axis.
    x_min : np.ndarray or None
        Minimum values of `x` across batch axis if `normalize` is True, otherwise None.
    x_max : np.ndarray or None
        Maximum values of `x` across batch axis if `normalize` is True, otherwise None.

    Raises
    ------
    ValueError
        If `x` does not have the correct dimensions.
    """

    # Convert the input to a Dask array if necessary.
    x = da.from_array(x)

    # Get the number of input dimensions.
    ndim = x.ndim

    if stack:
        nnormdim = 3
    else:
        nnormdim = 2

    # Check the number of dimensions.
    if ndim == nnormdim:
        batch_axis = False
    elif ndim == nnormdim + 1:
        batch_axis = True
    else:
        raise ValueError("Input does not have the correct dimensions.")

    # Normalize the input if necessary.
    if normalize:
        axis = tuple(range(ndim - nnormdim, ndim))
        stat_shape = (*x.shape[:-nnormdim], *((1, ) * nnormdim))
        x_min = np.min(x, axis=axis).reshape(stat_shape).compute()
        x_max = np.max(x, axis=axis).reshape(stat_shape).compute()
    else:
        x_min = None
        x_max = None

    return x, batch_axis, x_min, x_max
