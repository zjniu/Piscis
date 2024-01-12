import cv2 as cv
import jax
import jax.numpy as jnp
import numpy as np

from jax import jit, random, vmap
from scipy import ndimage
from typing import Any, List, Optional, Sequence, Tuple


class RandomAugment:

    """Transformer for random data augmentation.

    Attributes
    ----------
    output_size : Optional[Tuple[int, int]]
        Output size of the transformer.
    scales : List[float]
        List of image scales.
    dxys : List[Tuple[float, float]]
        List of image translation vectors.
    thetas : List[float]
        List of image rotation angles.
    affines : List[np.ndarray]
        List of affine transformation matrices.
    flips0 : List[bool]
        List of booleans for flipping along axis 0.
    flips1 : List[bool]
        List of booleans for flipping along axis 1.
    intensity_scales : List[float]
        List of image intensity scale factors.
    """

    def __init__(self) -> None:

        """Initialize a RandomAugment object."""

        self.output_size = None
        self.scales = []
        self.dxys = []
        self.thetas = []
        self.affines = []
        self.flips0 = []
        self.flips1 = []
        self.intensity_scales = []

    def generate_transforms(
            self,
            images: Sequence[np.ndarray],
            key: jax.Array,
            base_scales: Sequence[float],
            output_size: Tuple[int, int],
            min_scale_factor: float = 0.75,
            max_scale_factor: float = 1.25,
            max_intensity_scale_factor: float = 5
    ) -> None:

        """Generate random transformations.

        Parameters
        ----------
        images : Sequence[np.ndarray]
            Images to transform.
        key : jax.Array
            Random key used for generating random transformations.
        base_scales : Sequence[float]
            List of base scales for each image.
        output_size : Tuple[int, int]
            Output size of the transformer.
        min_scale_factor : float, optional
            Minimum scale factor. Default is 0.75.
        max_scale_factor : float, optional
            Maximum scale factor. Default is 1.25.
        max_intensity_scale_factor : float, optional
            Maximum intensity scale factor. Intensity scale factors are sampled from a log-uniform distribution with
            support on the interval [1 / `max_intensity_scale_factor`, `max_intensity_scale_factor`]. Default is 5.
        """

        # Reset class attributes.
        self.output_size = output_size
        self.flips0 = []
        self.flips1 = []
        self.scales = []
        self.dxys = []
        self.thetas = []
        self.affines = []
        self.intensity_scales = []

        for image, base_scale in zip(images, base_scales):

            # Random flips.
            key, *subkeys = random.split(key, 3)
            flip0 = random.uniform(subkeys[0]) > 0.5
            self.flips0.append(float(flip0))
            flip1 = random.uniform(subkeys[1]) > 0.5
            self.flips1.append(float(flip1))

            # Random scaling.
            key, subkey = random.split(key)
            scale = base_scale * (min_scale_factor + (max_scale_factor - min_scale_factor) * random.uniform(subkey))
            self.scales.append(float(scale))

            # Random translation.
            key, subkey = random.split(key)
            dxy = np.maximum(0, np.array([image.shape[1] * scale - output_size[1],
                                          image.shape[0] * scale - output_size[0]]))
            dxy = (random.uniform(subkey, (2, )) - 0.5) * dxy
            self.dxys.append(np.asarray(dxy))

            # Random rotation.
            key, subkey = random.split(key)
            theta = random.uniform(subkey) * 2 * np.pi
            self.thetas.append(float(theta))

            # Construct affine transformation.
            image_center = (image.shape[1] / 2, image.shape[0] / 2)
            affine = cv.getRotationMatrix2D(image_center, float(theta * 180 / np.pi), float(scale))
            affine[:, 2] += np.array(output_size) / 2 - np.array(image_center) + dxy
            self.affines.append(affine)

            # Random intensity scaling.
            key, subkey = random.split(key)
            intensity_scale = jnp.exp((random.uniform(subkey) - 0.5) * 2 * jnp.log(max_intensity_scale_factor))
            self.intensity_scales.append(float(intensity_scale))

    def apply_coord_transforms(
            self,
            coords: Sequence[np.ndarray],
            filter_coords: bool = True
    ) -> List[np.ndarray]:

        """Apply random transformations to coordinates.

        Parameters
        ----------
        coords : Sequence[np.ndarray]
            Coordinates to transform.
        filter_coords : bool, optional
            Whether to filter coordinates outside the transformed image. Default is True.

        Returns
        -------
        transformed_coords : List[np.ndarray]
            Transformed coordinates.
        """

        transformed_coords = []

        for coord, flip0, flip1, affine in zip(coords, self.flips0, self.flips1, self.affines):

            # Apply affine transformation
            coord = np.concatenate((np.flip(coord, axis=1), np.ones((len(coord), 1))), axis=1)
            transformed_coord = np.flip((coord @ affine.T), axis=1)

            # Random flip
            if flip0:
                transformed_coord[:, 0] = self.output_size[0] - 1 - transformed_coord[:, 0]
            if flip1:
                transformed_coord[:, 1] = self.output_size[1] - 1 - transformed_coord[:, 1]

            # Filter coordinates outside transformed image
            if filter_coords:
                transformed_coord = transformed_coord[np.all((transformed_coord > -0.5) &
                                                             (transformed_coord < np.array(self.output_size) - 0.5),
                                                             axis=1)]

            transformed_coords.append(transformed_coord)

        return transformed_coords

    def apply_image_transforms(
            self,
            images: Sequence[np.ndarray],
            interpolation: str = 'nearest'
    ) -> List[np.ndarray]:

        """Apply random transformations to images.

        Parameters
        ----------
        images : Sequence[np.ndarray]
            Images to transform.
        interpolation : str, optional
            Interpolation mode for image scaling. Supported modes are 'nearest' or 'bilinear'. Default is 'nearest'.

        Returns
        -------
        transformed_images : List[np.ndarray]
            Transformed images.

        Raises
        ------
        ValueError
            If the `interpolation` mode is not supported.
        """

        transformed_images = []

        for image, affine, flip0, flip1, intensity_scale in \
                zip(images, self.affines, self.flips0, self.flips1, self.intensity_scales):

            # Apply affine transformation
            if interpolation == 'nearest':
                image = cv.warpAffine(image, M=affine, dsize=self.output_size, flags=cv.INTER_NEAREST)
            elif interpolation == 'bilinear':
                image = cv.warpAffine(image, M=affine, dsize=self.output_size, flags=cv.INTER_LINEAR)
            else:
                raise ValueError("Interpolation mode is not supported.")

            # Random flip
            if flip0:
                image = np.flip(image, axis=0)
            if flip1:
                image = np.flip(image, axis=1)

            # Random intensity scaling
            image = image * intensity_scale

            transformed_images.append(image)

        return transformed_images


def batch_adjust(
        images: Sequence[np.ndarray],
        adjustment: Optional[str],
        **kwargs: Any
) -> Sequence[np.ndarray]:

    """Batch adjust images.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        Images to adjust.
    adjustment : Optional[str]
        Adjustment type. Supported types are 'normalize' and 'standardize'.
    **kwargs : Any
        Keyword arguments for the adjustment function.

    Returns
    -------
    adjusted_images : Sequence[np.ndarray]
        Adjusted images.
    """

    if adjustment is not None:
        images = list(images)
        adjusted_images = np.empty(len(images), dtype=object)
        for i, image in enumerate(images):
            adjusted_images[i] = adjust(image, adjustment, **kwargs)
    else:
        adjusted_images = images

    return adjusted_images


def adjust(
        image: np.ndarray,
        adjustment: Optional[str],
        **kwargs: Any
) -> np.ndarray:

    """Adjust an image.

    Parameters
    ----------
    image : np.ndarray
        Image to adjust.
    adjustment : Optional[str]
        Adjustment type. Supported types are 'normalize' and 'standardize'.
    **kwargs : Any
        Keyword arguments for the adjustment function.

    Returns
    -------
    adjusted_image : np.ndarray
        Adjusted image.

    Raises
    ------
    ValueError
        If the `adjustment` type is not supported.
    """

    if adjustment is None:
        adjusted_image = image
    elif adjustment == 'normalize':
        adjusted_image = normalize(image, **kwargs)
    elif adjustment == 'standardize':
        adjusted_image = standardize(image, **kwargs)
    else:
        raise ValueError("Adjustment type is not supported.")

    return adjusted_image


def normalize(
        image: np.ndarray,
        lower: float = 0,
        upper: float = 100,
        epsilon: float = 1e-7
) -> np.ndarray:

    """Normalize an image to the range [0, 1] based on the specified percentiles.

    Parameters
    ----------
    image : np.ndarray
        Image to normalize.
    lower : float, optional
        Lower percentile. Default is 0.
    upper : float, optional
        Upper percentile. Default is 100.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.

    Returns
    -------
    normalized_image : np.ndarray
        Normalized image.
    """

    image_lower = np.percentile(image, lower, axis=(-2, -1), keepdims=True)
    image_upper = np.percentile(image, upper, axis=(-2, -1), keepdims=True)
    normalized_image = (image - image_lower) / (image_upper - image_lower + epsilon)

    return normalized_image


def standardize(
        image: np.ndarray,
        epsilon: float = 1e-7
) -> np.ndarray:

    """Standardize an image to zero mean and unit variance.

    Parameters
    ----------
    image : np.ndarray
        Image to standardize.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.

    Returns
    -------
    standardized_image : np.ndarray
        Standardized image.
    """

    image_mean = image.mean(axis=(-2, -1), keepdims=True)
    image_std = image.std(axis=(-2, -1), keepdims=True)
    standardized_image = (image - image_mean) / (image_std + epsilon)

    return standardized_image


def voronoi_transform(
        coords: Sequence[np.ndarray],
        output_size: Tuple[int, int] = (256, 256),
        dilation_iterations: int = 1,
        coords_pad_length: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:

    """Transform a list of coordinates to generate ground truth binary labels and displacement vectors from each pixel
    to the nearest point via a Voronoi tessellation. Adapted from DeepCell Spots.

    Parameters
    ----------
    coords : Sequence[np.ndarray]
        List of coordinates.
    output_size : Tuple[int, int], optional
        Size of output arrays. Default is (256, 256).
    dilation_iterations : int, optional
        Number of iterations to dilate ground truth labels. Default is 1.
    coords_pad_length : Optional[int], optional
        Padded length of the coordinates sequence. Default is None.

    Returns
    -------
    deltas : jax.Array
        Array where each pixel is a vector to the nearest point in `coords`.
    labels : jax.Array
        Array where each pixel is a boolean for whether it contains a point in `coords`.

    References
    ----------
    .. [1] Laubscher, Emily, et al. "vanvalenlab/deepcell-spots: Deep Learning Library for Spot Detection." GitHub,
           https://github.com/vanvalenlab/deepcell-spots.
    """

    # Initialize the deltas and labels arrays.
    batch_size = len(coords)
    deltas = np.zeros((batch_size, *output_size, 2), dtype=float)
    labels = np.zeros((batch_size, *output_size), dtype=bool)

    # Determine the padded length of the coordinates sequence.
    coords_max_length = np.max([len(coord) for coord in coords])
    if (coords_pad_length is None) or (coords_pad_length < coords_max_length):
        coords_pad_length = coords_max_length

    # Generate ranges.
    i, j = np.arange(output_size[0]), np.arange(output_size[1])

    # Generate the dilation structuring element.
    structure = ndimage.generate_binary_structure(2, 2)

    for k, coord in enumerate(coords):

        # Remove coordinates outside the output arrays.
        coord = coord[(coord[:, 0] > -0.5) & (coord[:, 0] < output_size[0] - 0.5) &
                      (coord[:, 1] > -0.5) & (coord[:, 1] < output_size[1] - 0.5)]

        # Generate the labels array.
        rounded_coords = np.rint(coord).astype(int)
        labels[k][rounded_coords[:, 0], rounded_coords[:, 1]] = True

        # Apply the Euclidean distance transform on the labels array.
        edt_indices = ndimage.distance_transform_edt(~labels[k], return_distances=False, return_indices=True)

        # Pad coordinates to the same length.
        padding = ((0, coords_pad_length - len(coord)), (0, 0))
        coord = np.pad(coord, padding, constant_values=-1)

        # Apply the vectorized distance transform function.
        deltas[k] = vmap_vt(coord, edt_indices, i, j)

        # Dilate the labels array if necessary.
        if dilation_iterations > 0:
            labels[k] = ndimage.binary_dilation(labels[k], structure=structure, iterations=dilation_iterations)

    # Expand the shape of the labels array.
    labels = np.expand_dims(labels, axis=-1)

    return deltas, labels


@jit
def _vt(
        coord: jax.Array,
        edt_index: jax.Array,
        i: int,
        j: int
):

    """Voronoi transform for a single pixel.

    Parameters
    ----------
    coord : jax.Array
        List of coordinates.
    edt_index : jax.Array
        Euclidean distance transform index.
    i : int
        Pixel row index.
    j : int
        Pixel column index.

    Returns
    -------
    delta : jax.Array
        Vector to the nearest point.
    """

    distances = jnp.linalg.norm(coord - edt_index, axis=-1)
    delta = coord[jnp.argmin(distances)] - jnp.array((i, j))

    return delta


vmap_vt = vmap(vmap(_vt, in_axes=(None, 1, None, 0)), in_axes=(None, 1, 0, None))
