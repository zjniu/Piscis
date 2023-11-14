import dask.array as da
import deeptile
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from deeptile import lift, Output
from deeptile.core.data import Tiled
from deeptile.extensions import stitch
from flax import serialization
from functools import partial
from jax import jit
from jax._src import compilation_cache
from jax.lib import xla_bridge
from skimage.transform import resize
from typing import Dict, Optional, Sequence, Tuple, Union

from piscis import utils
from piscis.downloads import download_pretrained_model
from piscis.models.spots import round_input_size, SpotsModel
from piscis.paths import CACHE_DIR, MODELS_DIR


class Piscis:

    """Class for running the Piscis algorithm.

    Attributes
    ----------
    model_name : str
        Model name.
    batch_size : int
        Batch size for the CNN.
    model : SpotsModel
        Instance of the SpotsModel class.
    variables : Dict
        Model variables.
    adjustment : str
        Adjustment type applied to images during preprocessing.
    input_size : Tuple[int, int]
        Input size for the CNN.
    dilation_iterations : int
        Number of iterations used to dilate ground truth labels during training.
    apply : Callable
        Compiled model apply function.
    """

    def __init__(
            self,
            model_name: str = '20230905',
            batch_size: int = 4,
            cache: bool = True,
            input_size: Optional[Tuple[int, int]] = None
    ) -> None:

        """Initialize a Piscis object and compile the model.

        Parameters
        ----------
        model_name : str, optional
            Model name. Default is '20230905'.
        batch_size : int, optional
            Batch size for the CNN. Default is 4.
        cache : bool, optional
            Whether to use compilation cache. Default is True.
        input_size : Optional[Tuple[int, int]], optional
            Input size for the CNN. If None, it is obtained from the model dictionary. Default is None.
        """

        # Set the batch size to 1 if running on CPU.
        if xla_bridge.get_backend().platform == 'cpu':
            batch_size = 1

        # Initialize the compilation cache.
        if cache:
            compilation_cache.initialize_cache(CACHE_DIR)

        # Load the model.
        self.model_name = model_name
        self.batch_size = batch_size
        model_path = MODELS_DIR / model_name
        if not model_path.is_file():
            download_pretrained_model(model_name)
        with open(MODELS_DIR / model_name, 'rb') as f_model:
            model_dict = serialization.from_bytes(target=None, encoded_bytes=f_model.read())
            self.variables = model_dict['variables']
            self.adjustment = model_dict['adjustment']
            if input_size is None:
                input_size = model_dict['input_size']
                input_size = (input_size['0'], input_size['1'])
            else:
                input_size = round_input_size(input_size)
            self.input_size = input_size
            self.dilation_iterations = model_dict['dilation_iterations']

        # Define the model apply function.
        kernel_size = (2 * self.dilation_iterations + 1, ) * 2
        self.apply = partial(apply, variables=self.variables, kernel_size=kernel_size)

    def predict(
            self,
            x: Union[np.ndarray, da.Array],
            stack: bool = False,
            scale: float = 1.0,
            threshold: float = 1.0,
            min_distance: int = 1,
            intermediates: bool = False
    ) -> Union[Tuple[np.ndarray, xr.DataArray], np.ndarray]:

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
            Spot detection threshold. Can be interpreted as the minimum number of fully confident pixels necessary to
            identify a spot. Default is 1.0.
        min_distance : int, optional
            Minimum distance between spots. Default is 1.
        intermediates : bool, optional
            Whether to return intermediate feature maps. Default is False.

        Returns
        -------
        coords : np.ndarray
            Predicted spot coordinates.
        y : np.ndarray, optional
            Intermediate feature maps. Only returned if `intermediates` is True.
        """

        # Preprocess the input.
        x, batch_axis, x_stats = self._preprocess(x, stack)

        # Create a DeepTile object and get tiles.
        dt = deeptile.load(x, link_data=False, dask=True)
        tile_size = (round(self.input_size[0] / scale), round(self.input_size[1] / scale))
        scales = (np.array([self.input_size]) - 1) / (np.array(tile_size) - 1)
        tiles = dt.get_tiles(tile_size=tile_size, overlap=(0.1, 0.1)).pad(mode='symmetric')

        # Adjust tiles if necessary.
        if x_stats is not None:
            if self.adjustment == 'normalize':
                x_min, x_max = x_stats
                tiles = lift(lambda t: (t - x_min) / (x_max - x_min + 1e-7))(tiles)
            elif self.adjustment == 'standardize':
                x_mean, x_std = x_stats
                tiles = lift(lambda t: (t - x_mean) / (x_std + 1e-7))(tiles)

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

            dims = tuple(dim for cond, dim in ((batch_axis, 'n'), (stack, 'z')) if cond) + ('c', 'y', 'x')
            y = xr.DataArray(y, dims=dims)

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
            Spot detection threshold. Can be interpreted as the minimum number of fully confident pixels necessary to
            identify a spot.
        min_distance : int
            Minimum distance between spots.
        intermediates : bool
            Whether to return intermediate feature maps.

        Returns
        -------
        coords : Output
            Predicted spot coordinates.
        y : np.ndarray, optional
            Intermediate feature maps. Only returned if `intermediates` is True.
        """

        # Lift process and postprocess functions.
        process_stack = lift(partial(self._process, intermediates=intermediates),
                             vectorized=True, batch_axis=True, pad_final_batch=True, batch_size=self.batch_size)
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
                coords, postprocess_variables = \
                    postprocess_stack.init(carry[0], carry[1], scales, threshold, min_distance)

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
            y = np.concatenate((carry[0],
                                np.expand_dims(carry[2], axis=-1), np.expand_dims(carry[1], axis=-1)), axis=-1)
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
            Spot detection threshold. Can be interpreted as the minimum number of fully confident pixels necessary to
            identify a spot.
        min_distance : int
            Minimum distance between spots.
        intermediates : bool
            Whether to return intermediate feature maps.

        Returns
        -------
        coords : Output
            Predicted spot coordinates.
        y : np.ndarray, optional
            Intermediate feature maps. Only returned if `intermediates` is True.
        """

        # Process tiles.
        if intermediates:
            deltas, pooled_labels, labels = self._process(tiles, intermediates=intermediates)
        else:
            deltas, pooled_labels = self._process(tiles, intermediates=intermediates)
            labels = None

        # Postprocess tiles.
        coords = np.empty(len(deltas), dtype=object)
        for i, (d, pl) in enumerate(zip(deltas, pooled_labels)):
            coords[i] = utils.compute_spot_coordinates(d, pl, threshold=threshold, min_distance=min_distance)
            coords[i][:, -2:] = coords[i][:, -2:] / scales
        coords = Output(coords, isimage=False, stackable=False, tile_scales=(1.0, 1.0))

        if intermediates:
            y = np.concatenate(
                (deltas, np.expand_dims(labels, axis=-1), np.expand_dims(pooled_labels, axis=-1)), axis=-1
            )
            y = np.moveaxis(y, -1, 1)
            return coords, y
        else:
            return coords

    def _preprocess(
            self,
            x: Union[np.ndarray, da.Array],
            stack: bool
    ) -> Tuple[da.Array, bool, Optional[Tuple[np.ndarray, np.ndarray]]]:

        """Preprocess the input.

        Parameters
        ----------
        x : np.ndarray or dask.array.Array
            Image or stack of images.
        stack : bool
            Whether `x` is a stack of images.

        Returns
        -------
        x : Union[np.ndarray, da.Array]
            Preprocessed image or stack of images.
        batch_axis : bool
            Whether `x` has a batch axis.
        x_stats : Optional[Tuple[np.ndarray, np.ndarray]]
            Statistics of `x` across each plane used for image adjustment.

        Raises
        ------
        ValueError
            If `x` does not have the correct dimensions.
        """

        # Convert the input to a Dask array if necessary.
        if isinstance(x, xr.DataArray):
            x = x.data
        if not isinstance(x, da.Array):
            x = da.from_array(x)

        # Get the number of input dimensions.
        ndim = x.ndim

        # Check the number of dimensions.
        if ((ndim == 4) and stack) or ((ndim == 3) and (not stack)):
            batch_axis = True
        elif ((ndim == 2) and (not stack)) or ((ndim == 3) and stack):
            batch_axis = False
        else:
            raise ValueError("The input does not have the correct dimensions.")

        # Standardize the input if necessary.
        if self.adjustment == 'normalize':
            x_min = x.min(axis=(-2, -1), keepdims=True).compute()
            x_max = x.max(axis=(-2, -1), keepdims=True).compute()
            x_stats = (x_min, x_max)
        elif self.adjustment == 'standardize':
            x_mean = x.mean(axis=(-2, -1), keepdims=True).compute()
            x_std = x.std(axis=(-2, -1), keepdims=True).compute()
            x_stats = (x_mean, x_std)
        else:
            x_stats = None

        return x, batch_axis, x_stats

    def _process(
            self,
            tiles: Tiled,
            intermediates: bool
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

        """Process tiles.

        Parameters
        ----------
        tiles : Tiled
            Tiles of images.
        intermediates : bool
            Whether to keep intermediate feature maps.

        Returns
        -------
        deltas : np.ndarray
            Predicted displacement vectors.
        pooled_labels : np.ndarray
            Predicted pooled labels.
        labels : np.ndarray, optional
            Predicted binary labels. Only returned if `intermediates` is True.
        """

        # Compute Dask arrays if necessary.
        tiles = tiles.compute()

        # Resize tiles if necessary.
        if tiles.shape[1:3] != self.input_size:
            tiles = resize(tiles, (tiles.shape[0], *self.input_size), preserve_range=True)

        # Run the jitted model apply function on tiles.
        if intermediates:
            deltas, pooled_labels, labels = self.apply(tiles)
            deltas = np.asarray(deltas)
            pooled_labels = np.asarray(pooled_labels)
            labels = np.asarray(labels)
            return deltas, pooled_labels, labels
        else:
            deltas, pooled_labels, _ = self.apply(tiles)
            deltas = np.asarray(deltas)
            pooled_labels = np.asarray(pooled_labels)
            return deltas, pooled_labels

    @staticmethod
    def _postprocess_stack(
            deltas: np.ndarray,
            pooled_labels: np.ndarray,
            scales: np.ndarray,
            threshold: float,
            min_distance: int
    ) -> Output:

        """Postprocess tiles.

        Parameters
        ----------
        deltas : np.ndarray
            Predicted displacement vectors.
        pooled_labels : np.ndarray
            Predicted pooled labels.
        scales : np.ndarray
            Scales for rescaling tiles.
        threshold : float
            Spot detection threshold. Can be interpreted as the minimum number of fully confident pixels necessary to
            identify a spot.
        min_distance : int
            Minimum distance between spots.

        Returns
        -------
        coords : Output
            Predicted spot coordinates.
        """

        # Compute spot coordinates.
        coords = utils.compute_spot_coordinates(deltas, pooled_labels, threshold=threshold, min_distance=min_distance)

        # Rescale coordinates.
        coords[:, -2:] = coords[:, -2:] / scales
        coords = Output(coords, isimage=False, stackable=False, tile_scales=(1.0, 1.0))

        return coords


@partial(jit, static_argnums=2)
def apply(
        x: jax.Array,
        variables: Dict,
        kernel_size: Sequence[int]
) -> Tuple[jax.Array, jax.Array, jax.Array]:

    """Apply SpotsModel to a batch images.

    x : jax.Array
        Batch of images.
    variables : Dict
        Model variables.
    kernel_size : Sequence[int]
        Kernel size or window size of the sum pooling operation.

    Returns
    -------
    deltas : jax.Array
        Predicted displacements vectors.
    pooled_labels : jax.Array
        Predicted pooled labels.
    labels : jax.Array
        Predicted binary labels.
    """

    x = jnp.expand_dims(x, axis=-1)
    deltas, labels = SpotsModel().apply(variables, x, False)
    labels = labels[:, :, :, 0]
    pooled_labels = utils.vmap_sum_pool(deltas, labels, kernel_size)

    return deltas, pooled_labels, labels


def adjust_parameters(
        y: xr.DataArray,
        threshold: float = 1.0,
        min_distance: int = 1
) -> np.ndarray:
    """Adjust tunable parameters for a given set of intermediate feature maps.

    Parameters
    ----------
    y : xr.DataArray
        Intermediate feature maps.
    threshold: float
        Spot detection threshold. Can be interpreted as the minimum number of fully confident pixels necessary to
        identify a spot. Default is 1.0.
    min_distance : int, optional
        Minimum distance between spots. Default is 1.

    Returns
    -------
    coords : np.ndarray
        Predicted spot coordinates.
    """

    if 'n' in y.dims:

        batch_axis_len = y.shape[0]
        coords = np.empty(batch_axis_len, dtype=object)
        for i in range(batch_axis_len):
            deltas = np.moveaxis(y[i, ..., :2, :, :].to_numpy(), -3, -1)
            pooled_labels = y[i, ..., 3, :, :].to_numpy()
            coords[i] = utils.compute_spot_coordinates(deltas, pooled_labels,
                                                       threshold=threshold, min_distance=min_distance)

    else:

        deltas = np.moveaxis(y[..., :2, :, :].to_numpy(), -3, -1)
        pooled_labels = y[..., 3, :, :].to_numpy()
        coords = utils.compute_spot_coordinates(deltas, pooled_labels, threshold=threshold, min_distance=min_distance)

    return coords
