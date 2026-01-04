import dask.array as da
import deeptile
import numpy as np
import torch
import xarray as xr

from deeptile import lift, Output
from deeptile.core.data import Tiled
from deeptile.extensions import stitch
from skimage.transform import resize
from typing import Optional, Tuple, Union

from piscis.convert import convert_jax_to_torch_state_dict
from piscis.downloads import download_pretrained_model
from piscis.models.spots import round_input_size, SpotsModel
from piscis.paths import MODELS_DIR
from piscis.utils import compute_spot_coordinates


class _Piscis:

    """Base class for running the Piscis algorithm.

    Attributes
    ----------
    model_name : str
        Model name.
    batch_size : int
        Batch size for the CNN.
    adjustment : str
        Adjustment type applied to images during preprocessing.
    input_size : Tuple[int, int]
        Input size for the CNN.
    dilation_iterations : int
        Number of iterations used to dilate ground truth labels during training.
    channels : int
        Number of channels in the input images.
    device : Optional[Union[str, torch.device]]
        Device to run the model on.
    pooling : str
        Pooling type applied to labels.
    model : SpotsModel
        Model.
    """

    def __init__(
            self,
            model_name: str,
            batch_size: int,
            input_size: Optional[Tuple[int, int]],
            device: Optional[Union[str, torch.device]],
            pooling: str
    ) -> None:

        """Initialize a Piscis object.

        Parameters
        ----------
        model_name : str
            Model name.
        batch_size : int
            Batch size for the CNN.
        input_size : Optional[Tuple[int, int]]
            Input size for the CNN. If None, it is obtained from the model dictionary.
        device : Optional[Union[str, torch.device]]
            Device to run the model on.
        pooling : str
            Pooling type applied to labels.
        """

        # Load the model.
        self.model_name = model_name
        self.batch_size = batch_size
        model_path = MODELS_DIR / f'{model_name}.pt'
        if not model_path.is_file():
            if (MODELS_DIR / model_name).is_file():
                state_dict = convert_jax_to_torch_state_dict(model_name)
                torch.save(state_dict, model_path)
            else:
                download_pretrained_model(model_name)
        with open(model_path, 'rb') as f_model:
            state_dict = torch.load(f_model, map_location='cpu', weights_only=False)
            metadata = state_dict.pop('metadata')
            if 'metrics_log' in metadata:
                self.metrics_log = metadata['metrics_log']
            self.adjustment = metadata['adjustment']
            if input_size is None:
                input_size = metadata['input_size']
            else:
                input_size = round_input_size(input_size)
            self.input_size = input_size
            self.dilation_iterations = metadata['dilation_iterations']
            self.channels = metadata['channels']
        self.device = device
        self.pooling = pooling
        kernel_size = (2 * self.dilation_iterations + 1, ) * 2
        self.model = SpotsModel(in_channels=self.channels, pooling=self.pooling, kernel_size=kernel_size)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)

    def _predict(
            self,
            x: Union[np.ndarray, da.Array],
            stack: bool,
            scale: float,
            threshold: float,
            min_distance: int,
            intermediates: bool
    ) -> Union[Tuple[np.ndarray, xr.DataArray], np.ndarray]:

        """Predict spots in an image or stack of images.

        Parameters
        ----------
        x : np.ndarray or da.Array
            Image or stack of images.
        stack : bool, optional
            Whether `x` is a stack of images.
        scale : float, optional
            Scale factor for rescaling `x`.
        threshold : float, optional
            Spot detection threshold.
        min_distance : int, optional
            Minimum distance between spots.
        intermediates : bool, optional
            Whether to return intermediate feature maps.

        Returns
        -------
        coords : np.ndarray
            Predicted spot coordinates.
        y : np.ndarray, optional
            Intermediate feature maps. Only returned if `intermediates` is True.
        """

        # Preprocess the input.
        x, batch_axis = self._preprocess(x, stack)

        # Create a DeepTile object and get tiles.
        dt = deeptile.load(x, link_data=False, dask=True)
        tile_size = (round(self.input_size[0] / scale), round(self.input_size[1] / scale))
        scales = (np.array([self.input_size]) - 1) / (np.array(tile_size) - 1)
        if x.shape[-2] > tile_size[0]:
            overlap_i = 0.1
        else:
            overlap_i = 0
        if x.shape[-1] > tile_size[1]:
            overlap_j = 0.1
        else:
            overlap_j = 0
        if (2 * x.shape[-2] >= tile_size[0]) and (2 * x.shape[-1] >= tile_size[1]):
            pad_mode = 'symmetric'
        else:
            pad_mode = 'constant'
        tiles = dt.get_tiles(tile_size=tile_size, overlap=(overlap_i, overlap_j)).pad(mode=pad_mode)

        # Predict spots in tiles and stitch.
        if stack and batch_axis:

            batch_axis_len = x.shape[0]
            stack_axis_len = x.shape[-4]
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
                stack_axis_len = x.shape[-4]
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
        process_stack = lift(self._process,
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
                    carry[-2][i, j] = None
                    carry[-1][i, j] = None
                mod = mod - stack_axis_len
                if j == coords.shape[1] - 1:
                    i = i + 1
                    j = 0
                else:
                    j = j + 1

        if intermediates:
            y = np.concatenate((np.expand_dims(carry[0], axis=-3), carry[1]), axis=-3)
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
        labels, deltas = self._process(tiles)

        # Postprocess tiles.
        coords = np.empty(len(deltas), dtype=object)
        for i, (l, d) in enumerate(zip(labels, deltas)):
            coords[i] = compute_spot_coordinates(l, d, threshold=threshold, min_distance=min_distance)
            coords[i][:, -2:] = coords[i][:, -2:] / scales
        coords = Output(coords, isimage=False, stackable=False, tile_scales=(1.0, 1.0))

        if intermediates:
            y = np.concatenate((np.expand_dims(labels, axis=-3), deltas), axis=-3)
            return coords, y
        else:
            return coords

    def _preprocess(
            self,
            x: Union[np.ndarray, da.Array],
            stack: bool
    ) -> Tuple[da.Array, bool]:

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

        Raises
        ------
        ValueError
            If `x` does not have the correct dimensions.
        ValueError
            If `x` does not have the correct number of channels.
        """

        # Check the channel axis.
        if self.channels == 1:
            x = np.expand_dims(x, axis=-3)
        else:
            if x.shape[-3] != self.channels:
                raise ValueError("The input does not have the correct number of channels.")

        # Get the number of input dimensions.
        ndim = x.ndim

        # Check the number of dimensions.
        if ((ndim == 5) and stack) or ((ndim == 4) and (not stack)):
            batch_axis = True
        elif ((ndim == 3) and (not stack)) or ((ndim == 4) and stack):
            batch_axis = False
        else:
            raise ValueError("The input does not have the correct dimensions.")

        # Convert the input to a Dask array if necessary.
        if isinstance(x, xr.DataArray):
            x = x.data
        if not isinstance(x, da.Array):
            chunks = []
            if batch_axis:
                chunks.append(self.batch_size)
            if stack:
                chunks.append(1)
            chunks = chunks + [self.channels, *self.input_size]
            x = da.from_array(x, chunks=chunks)

        return x, batch_axis

    def _process(self, tiles: Tiled) -> Tuple[np.ndarray, np.ndarray]:

        """Process tiles.

        Parameters
        ----------
        tiles : Tiled
            Tiles of images.

        Returns
        -------
        labels : np.ndarray
            Predicted labels.
        deltas : np.ndarray
            Predicted displacement vectors.
        """

        # Adjust tiles if necessary.
        tiles = tiles.compute()
        if self.adjustment == 'normalize':
            x_min = tiles.min(axis=(-2, -1), keepdims=True)
            x_max = tiles.max(axis=(-2, -1), keepdims=True)
            tiles = (tiles - x_min) / (x_max - x_min + 1e-7)
        elif self.adjustment == 'standardize':
            x_mean = tiles.mean(axis=(-2, -1), keepdims=True)
            x_std = tiles.std(axis=(-2, -1), keepdims=True)
            tiles = (tiles - x_mean) / (x_std + 1e-7)

        # Resize tiles if necessary.
        if tiles.shape[-2:] != self.input_size:
            tiles = resize(tiles, (*tiles.shape[:-2], *self.input_size), preserve_range=True)

        # Run the model on tiles.
        labels, deltas = self._apply(tiles)
        
        return labels, deltas

    @staticmethod
    def _postprocess_stack(
            labels: np.ndarray,
            deltas: np.ndarray,
            scales: np.ndarray,
            threshold: float,
            min_distance: int
    ) -> Output:

        """Postprocess tiles.

        Parameters
        ----------
        labels : np.ndarray
            Predicted labels.
        deltas : np.ndarray
            Predicted displacement vectors.
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
        coords = compute_spot_coordinates(labels, deltas, threshold=threshold, min_distance=min_distance)

        # Rescale coordinates.
        coords[:, -2:] = coords[:, -2:] / scales
        coords = Output(coords, isimage=False, stackable=False, tile_scales=(1.0, 1.0))

        return coords
    

    def _apply(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        """Apply SpotsModel to a batch images.

        x : torch.Tensor
            Batch of images.

        Returns
        -------
        labels : np.ndarray
            Predicted binary labels.
        deltas : np.ndarray
            Predicted displacements vectors.
        """

        x = torch.from_numpy(x.astype(np.float32, copy=False)).to(self.device)
        with torch.inference_mode():
            labels, deltas = self.model(x)

        labels = labels.cpu().numpy()
        deltas = deltas.cpu().numpy()

        return labels, deltas
    

class Piscis(_Piscis):

    """Class for running the Piscis algorithm.

    Attributes
    ----------
    model_name : str
        Model name.
    batch_size : int
        Batch size for the CNN.
    adjustment : str
        Adjustment type applied to images during preprocessing.
    input_size : Tuple[int, int]
        Input size for the CNN.
    dilation_iterations : int
        Number of iterations used to dilate ground truth labels during training.
    channels : int
        Number of channels in the input images.
    device : Optional[Union[str, torch.device]]
        Device to run the model on.
    pooling : str
        Pooling type applied to labels.
    model : SpotsModel
        Model.
    """

    def __init__(
            self,
            model_name: str = '20230905',
            batch_size: int = 1,
            input_size: Optional[Tuple[int, int]] = None,
            device: Optional[Union[str, torch.device]] = None
    ) -> None:

        """Initialize a Piscis object.

        Parameters
        ----------
        model_name : str, optional
            Model name. Default is '20230905'.
        batch_size : int, optional
            Batch size for the CNN. Default is 1.
        input_size : Optional[Tuple[int, int]], optional
            Input size for the CNN. If None, it is obtained from the model dictionary. Default is None.
        device : Optional[Union[str, torch.device]], optional
            Device to run the model on. Default is None.
        """

        super().__init__(model_name=model_name, batch_size=batch_size, input_size=input_size, device=device,
                         pooling='max')
        
    def predict(
            self,
            x: Union[np.ndarray, da.Array],
            stack: bool = False,
            scale: float = 1.0,
            threshold: float = 0.5,
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
            Spot detection threshold. Default is 0.5.
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

        pred = self._predict(x, stack, scale, threshold, min_distance, intermediates)
        if intermediates:
            coords, y = pred
            return coords, y
        else:
            coords = pred
            return coords
        

class PiscisLegacy(_Piscis):

    """Class for running the PyTorch port of the legacy Piscis algorithm originally implemented in JAX.

    Attributes
    ----------
    model_name : str
        Model name.
    batch_size : int
        Batch size for the CNN.
    adjustment : str
        Adjustment type applied to images during preprocessing.
    input_size : Tuple[int, int]
        Input size for the CNN.
    dilation_iterations : int
        Number of iterations used to dilate ground truth labels during training.
    channels : int
        Number of channels in the input images.
    device : Optional[Union[str, torch.device]]
        Device to run the model on.
    pooling : str
        Pooling type applied to labels.
    model : SpotsModel
        Model.
    """

    def __init__(
            self,
            model_name: str = '20230905',
            batch_size: int = 1,
            input_size: Optional[Tuple[int, int]] = None,
            device: Optional[Union[str, torch.device]] = None
    ) -> None:

        """Initialize a Piscis object.

        Parameters
        ----------
        model_name : str, optional
            Model name. Default is '20230905'.
        batch_size : int, optional
            Batch size for the CNN. Default is 1.
        input_size : Optional[Tuple[int, int]], optional
            Input size for the CNN. If None, it is obtained from the model dictionary. Default is None.
        device : Optional[Union[str, torch.device]], optional
            Device to run the model on. Default is None.
        """

        super().__init__(model_name=model_name, batch_size=batch_size, input_size=input_size, device=device,
                         pooling='sum')
        
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

        pred = self._predict(x, stack, scale, threshold, min_distance, intermediates)
        if intermediates:
            coords, y = pred
            return coords, y
        else:
            coords = pred
            return coords


def adjust_parameters(
        y: xr.DataArray,
        threshold: float = 0.5,
        min_distance: int = 1
) -> np.ndarray:
    """Adjust tunable parameters for a given set of intermediate feature maps.

    Parameters
    ----------
    y : xr.DataArray
        Intermediate feature maps.
    threshold: float
        Spot detection threshold. Default is 0.5.
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
            labels = y[i, ..., 0, :, :].to_numpy()
            deltas = y[i, ..., 1:, :, :].to_numpy()
            coords[i] = compute_spot_coordinates(labels, deltas, threshold=threshold, min_distance=min_distance)

    else:

        labels = y[..., 0, :, :].to_numpy()
        deltas = y[..., 1:, :, :].to_numpy()
        coords = compute_spot_coordinates(labels, deltas, threshold=threshold, min_distance=min_distance)

    return coords
