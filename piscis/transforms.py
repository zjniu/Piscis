import cv2 as cv
import numpy as np
import torch

from scipy import ndimage
from typing import Any, Optional, Sequence, Tuple


class RandomAugment:

    """Transformer for random data augmentation.

    Attributes
    ----------
    rng : np.random.Generator
        Random number generator used for generating random transformations.
    output_size : Optional[Tuple[int, int]]
        Output size.
    flips0 : List[bool]
        List of booleans for flipping along axis 0.
    flips1 : List[bool]
        List of booleans for flipping along axis 1.
    scales : List[float]
        List of image scales.
    dxys : List[Tuple[float, float]]
        List of image translation vectors.
    thetas : List[float]
        List of image rotation angles.
    affines : List[np.ndarray]
        List of affine transformation matrices.
    intensity_shifts : List[float]
        List of image intensity shift values.
    intensity_scales : List[float]
        List of image intensity scale factors.
    """

    def __init__(
            self,
            seed: int,
            output_size: Tuple[int, int],
            augment: bool = True
    ) -> None:

        """Initialize a RandomAugment object.
        
        Parameters
        ----------
        seed : int
            Random seed used for generating random transformations.
        output_size : Tuple[int, int]
            Output size.
        augment : bool, optional
            Whether to apply random augmentations. Default is True.
        """

        self.rng = np.random.default_rng(seed)
        self.output_size = output_size
        self.augment = augment

    def apply(
            self,
            image: np.ndarray,
            coords: np.ndarray,
            filter_coords: bool = True,
            min_scale_factor: float = 0.75,
            max_scale_factor: float = 1.25,
            max_intensity_shift: float = 0.1,
            max_intensity_scale_factor: float = 5
    ) -> Tuple[np.ndarray, np.ndarray]:

        """Apply random transformations to an image and its corresponding coordinates.

        Parameters
        ----------
        images : np.ndarray
            Images to transform.
        coords : np.ndarray
            Coordinates to transform.
        filter_coords : bool, optional
            Whether to filter coordinates outside the transformed image. Default is True.
        min_scale_factor : float, optional
            Minimum scale factor. Default is 0.75.
        max_scale_factor : float, optional
            Maximum scale factor. Default is 1.25.
        max_intensity_shift : float, optional
            Maximum intensity shift. Intensity shifts are sampled from a uniform distribution with support on the
            interval [-`max_intensity_shift`, `max_intensity_shift`]. Default is 0.1.
        max_intensity_scale_factor : float, optional
            Maximum intensity scale factor. Intensity scale factors are sampled from a log-uniform distribution with
            support on the interval [1 / `max_intensity_scale_factor`, `max_intensity_scale_factor`]. Default is 5.
        
        Returns
        -------
        image : np.ndarray
            Transformed image.
        coords : np.ndarray
            Transformed coordinates.
        """

        # Determine the number of channels.
        if image.ndim == 2:
            channels = 1
        elif image.ndim == 3:
            channels = image.shape[0]
            image = np.moveaxis(image, 0, -1)

        # Random affine transformation.
        if self.augment:
            do_affine = self.rng.uniform() > 0.5
        else:
            do_affine = False

        if do_affine:

            # Random scaling.
            scale = self.rng.uniform(min_scale_factor, max_scale_factor)

            # Random translation.
            dxy = np.maximum(0, np.array([image.shape[1] * scale - self.output_size[1],
                                          image.shape[0] * scale - self.output_size[0]]))
            dxy = self.rng.uniform(-dxy / 2, dxy / 2, size=2)

            # Random rotation.
            theta = 2 * np.pi * self.rng.uniform()

        else:

            scale = 1
            dxy = np.array([0, 0])
            theta = self.rng.choice((0, np.pi / 2, np.pi, 3 * np.pi / 2))

        image_center = (image.shape[1] / 2, image.shape[0] / 2)
        affine = cv.getRotationMatrix2D(image_center, float(theta * 180 / np.pi), float(scale))
        affine[:, 2] += np.array(self.output_size) / 2 - np.array(image_center) + dxy
        image = cv.warpAffine(image, M=affine, dsize=self.output_size, flags=cv.INTER_LINEAR)
        coords = np.concatenate((np.flip(coords, axis=1), np.ones((len(coords), 1))), axis=1)
        coords = np.flip((coords @ affine.T), axis=1)

        # Add or move channels axis if necessary.
        if image.ndim == 2:
            image = image[None]
        elif image.ndim == 3:
            image = np.moveaxis(image, -1, 0)

        if self.augment:

            # Random flips.
            if self.rng.uniform() > 0.5:
                image = np.flip(image, axis=1)
                coords[:, 0] = self.output_size[0] - 1 - coords[:, 0]
            if self.rng.uniform() > 0.5:
                image = np.flip(image, axis=2)
                coords[:, 1] = self.output_size[1] - 1 - coords[:, 1]

            # Random intensity shift and scaling.
            if self.rng.uniform() > 0.5:
                intensity_shift = self.rng.uniform(-max_intensity_shift, max_intensity_shift, size=(channels, 1, 1))
                log_max_intensity_scale_factor = np.log(max_intensity_scale_factor)
                intensity_scale = np.exp(
                    self.rng.uniform(-log_max_intensity_scale_factor,
                                    log_max_intensity_scale_factor, size=(channels, 1, 1))
                )
                image = intensity_shift + intensity_scale * image

        # Filter coordinates outside transformed image
        if filter_coords:
            coords = coords[np.all((coords > -0.5) & (coords < np.array(self.output_size) - 0.5), axis=1)]

        return image, coords


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

    adjusted_images = np.empty(len(images), dtype=object)

    if adjustment is not None:
        for i, image in enumerate(images):
            adjusted_images[i] = adjust(image, adjustment, **kwargs)
    else:
        adjusted_images[:] = images

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


def batch_voronoi_transform(
        coords: Sequence[np.ndarray],
        output_size: Tuple[int, int] = (256, 256),
        dilation_iterations: int = 1,
        device: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """Batch Voronoi transform.

    Parameters
    ----------
    coords : Sequence[np.ndarray]
        List of coordinates.
    output_size : Tuple[int, int], optional
        Output size. Default is (256, 256).
    dilation_iterations : int, optional
        Number of iterations to dilate ground truth labels. Default is 1.
    device : Optional[str], optional
        Desired device of returned tensors. Default is None.

    Returns
    -------
    labels : torch.Tensor
        Tensor where each pixel is a boolean for whether it contains a point in `coords`.
    deltas : torch.Tensor
        Tensor where each pixel is a vector to the nearest point in `coords`.
    """

    # Initialize lists for labels and deltas.
    labels = []
    deltas = []

    # Generate index map.
    ii, jj = torch.meshgrid(
            torch.arange(output_size[0], dtype=torch.float, device=device),
            torch.arange(output_size[1], dtype=torch.float, device=device),
            indexing='ij'
        )
    index_map = torch.stack((ii, jj), dim=-1).view(1, -1, 2)

    # Apply the Voronoi transform.
    for c in coords:
        batch_labels, batch_deltas = voronoi_transform(c, output_size, dilation_iterations, device, index_map)
        labels.append(batch_labels)
        deltas.append(batch_deltas)

    # Stack the labels and deltas.
    labels = torch.stack(labels, dim=0)
    deltas = torch.stack(deltas, dim=0)

    return labels, deltas


def voronoi_transform(
        coords: np.ndarray,
        output_size: Tuple[int, int] = (256, 256),
        dilation_iterations: int = 1,
        device: Optional[str] = None,
        index_map: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:

    """Transform a list of coordinates to generate ground truth binary labels and displacement vectors from each pixel
    to the nearest point via a Voronoi tessellation. Adapted from DeepCell Spots.

    Parameters
    ----------
    coords : np.ndarray
        List of coordinates.
    output_size : Tuple[int, int], optional
        Output size. Default is (256, 256).
    dilation_iterations : int, optional
        Number of iterations to dilate ground truth labels. Default is 1.
    device : Optional[str], optional
        Desired device of returned tensors. Default is None.
    index_map : Optional[torch.Tensor], optional
        Precomputed index map. Will be generated internally if `index_map` is None. Default is None.

    Returns
    -------
    labels : torch.Tensor
        Tensor where each pixel is a boolean for whether it contains a point in `coords`.
    deltas : torch.Tensor
        Tensor where each pixel is a vector to the nearest point in `coords`.

    References
    ----------
    .. [1] Laubscher, Emily, et al. "vanvalenlab/deepcell-spots: Deep Learning Library for Spot Detection." GitHub,
           https://github.com/vanvalenlab/deepcell-spots.
    """
    
    # Generate the labels tensor.
    rounded_coords = np.rint(coords).astype(int)
    sorted_indices = np.lexsort((np.linalg.norm(coords - rounded_coords, axis=1), rounded_coords[:, 0], rounded_coords[:, 1]))
    coords = coords[sorted_indices]
    rounded_coords = rounded_coords[sorted_indices]
    labels = np.zeros(output_size, dtype=bool)
    labels[rounded_coords[:, 0], rounded_coords[:, 1]] = True
    if dilation_iterations > 0:
        labels = ndimage.binary_dilation(labels, structure=ndimage.generate_binary_structure(2, 2),
                                                 iterations=dilation_iterations)
    labels = torch.tensor(labels, dtype=torch.bool, device=device)

    # Convert coords to a tensor.
    coords = torch.tensor(coords, dtype=torch.float, device=device)
    rounded_coords = torch.tensor(rounded_coords, dtype=torch.float, device=device)

    # Generate index map if not provided.
    if index_map is None:
        ii, jj = torch.meshgrid(
            torch.arange(output_size[0], dtype=torch.float, device=device),
            torch.arange(output_size[1], dtype=torch.float, device=device),
            indexing='ij'
        )
        index_map = torch.stack((ii, jj), dim=-1).view(1, -1, 2)

    # Compute pairwise distances between each pixel and coordinate.
    distances = torch.cdist(index_map, rounded_coords[None])[0]
    nearest_coords = coords[torch.argmin(distances, dim=1)]

    # Compute the deltas tensor.
    deltas = (nearest_coords - index_map).view(output_size[0], output_size[1], 2).permute(2, 0, 1)

    return labels, deltas
