import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from scipy.optimize import least_squares
from scipy.spatial import KDTree
from typing import Dict, List, Tuple


@dataclass
class FitResult:

    """Fit result for a single particle.
    
    Attributes
    ----------
    i : int
        Particle index.
    x : float
        Fitted x-coordinate of the particle center.
    y : float
        Fitted y-coordinate of the particle center.
    amplitude : float
        Fitted amplitude of the particle.
    sigma : float
        Fitted standard deviation of the particle.
    background : float
        Fitted background level.
    residual_rms : float
        Root mean square of the fit residuals.
    success : bool
        Whether the fit was successful.
    """

    i: int
    x: float
    y: float
    amplitude: float
    sigma: float
    background: float
    residual_rms: float
    success: bool


def fit_coords(
        image: np.ndarray,
        coords: np.ndarray,
        cluster_threshold: float = 4.0,
        patch_padding: int = 4,
        sigma_init: float = 1.0,
        amplitude_rel_bounds: Tuple[float, float] = (0.1, 2.0),
        sigma_bounds: Tuple[float, float] = (0.5, 3.0),
) -> List[FitResult]:

    """Fit particle coordinates with a Gaussian mixture model.

    Parameters
    ----------
    image : np.ndarray
        Image.
    coords : np.ndarray
        Coordinates.
    cluster_threshold : float, optional
        Distance threshold for grouping particles into clusters. Default is 4.0.
    patch_padding : int, optional
        Number of pixels to pad the patch around each cluster of particles. Default is 4.
    sigma_init : float, optional
        Initial guess for the standard deviation of the Gaussians. Default is 1.0.
    amplitude_rel_bounds : Tuple[float, float], optional
        Bounds for the amplitude of the Gaussians relative to the initial guess. Default is (0.1, 2.0).
    sigma_bounds : Tuple[float, float], optional
        Bounds for the standard deviation of the Gaussians. Default is (0.5, 3.0).

    Returns
    -------
    results : List[FitResult]
        List of fit results for each particle.
    """

    # Group particles into clusters.
    clusters = _group_clusters(coords, cluster_threshold)

    # Fit each cluster of particles with a Gaussian mixture model.
    results = []
    for cluster in clusters:
        results.extend(
            _fit_cluster(
                image, cluster, coords, patch_padding,
                sigma_init, amplitude_rel_bounds, sigma_bounds
            )
        )

    return results


def remove_duplicate_coords(
        coords: np.ndarray,
        cluster_threshold: float = 1.0
) -> np.ndarray:

    """Remove duplicate coordinates within a cluster threshold.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates.
    cluster_threshold : float, optional
        Distance threshold for grouping particles into clusters. Default is 1.0.

    Returns
    -------
    new_coords : np.ndarray
        Coordinates without duplicates.
    """

    # Group particles into clusters.
    clusters = _group_clusters(coords, cluster_threshold)

    # Compute the mean coordinates of each cluster to remove duplicates.
    new_coords = []
    for cluster in clusters:
        new_coords.append(np.mean(coords[cluster], axis=0))
    if new_coords:
        new_coords = np.array(new_coords)
    else:
        new_coords = np.empty((0, 2))

    return new_coords


def _fit_cluster(
        image: np.ndarray,
        cluster: List[int],
        coords: np.ndarray,
        patch_padding: int,
        sigma_init: float,
        amplitude_rel_bounds: Tuple[float, float],
        sigma_bounds: Tuple[float, float],
) -> List[FitResult]:

    """Fit a cluster of particles with a Gaussian mixture model.

    Parameters
    ----------
    image : np.ndarray
        Image.
    cluster : List[int]
        List of particle indices in the cluster.
    coords : np.ndarray
        Coordinates.
    patch_padding : int
        Number of pixels to pad the patch around the cluster.
    sigma_init : float
        Initial guess for the standard deviation of the Gaussians.
    amplitude_rel_bounds : Tuple[float, float]
        Bounds for the amplitude of the Gaussians relative to the initial guess.
    sigma_bounds : Tuple[float, float]
        Bounds for the standard deviation of the Gaussians.

    Returns
    -------
    results : List[FitResult]
        List of fit results for each particle in the cluster.
    """

    # Extract the patch around the cluster of particles.
    n_particles = len(cluster)
    ys = coords[cluster, 0]
    xs = coords[cluster, 1]
    y_min = max(0, round(ys.min()) - patch_padding)
    y_max = min(image.shape[0], round(ys.max()) + patch_padding + 1)
    x_min = max(0, round(xs.min()) - patch_padding)
    x_max = min(image.shape[1], round(xs.max()) + patch_padding + 1)

    patch = image[y_min:y_max, x_min:x_max].astype(float, copy=False)
    y = np.arange(patch.shape[0], dtype=float)
    x = np.arange(patch.shape[1], dtype=float)
    xx, yy = np.meshgrid(x, y)

    # Initialize parameters and set bounds for the optimization.
    params_init = []
    lower_bounds = []
    upper_bounds = []

    for i in cluster:
        c_y, c_x = coords[i]
        i_y = np.clip(round(c_y), 0, image.shape[0] - 1)
        i_x = np.clip(round(c_x), 0, image.shape[1] - 1)
        amplitude_init = image[i_y, i_x]
        x_init = c_x - x_min
        y_init = c_y - y_min
        params_init.extend((amplitude_init, x_init, y_init, sigma_init))
        lower_bounds.extend((amplitude_rel_bounds[0] * amplitude_init,
                             x_init - sigma_init, y_init - sigma_init, sigma_bounds[0]))
        upper_bounds.extend((amplitude_rel_bounds[1] * amplitude_init + 1e-7,
                             x_init + sigma_init, y_init + sigma_init, sigma_bounds[1]))

    params_init.append(0.)
    lower_bounds.append(0.)
    upper_bounds.append(amplitude_rel_bounds[1] * patch.max())

    params_init = np.array(params_init, dtype=float)
    lower_bounds = np.array(lower_bounds, dtype=float)
    upper_bounds = np.array(upper_bounds, dtype=float)

    # Fit the Gaussian mixture model using least squares optimization.
    objective = GaussianMixtureObjective(n_particles, xx, yy, patch)
    sol = least_squares(
        objective.compute_residuals,
        params_init,
        jac=objective.compute_jacobian,
        bounds=(lower_bounds, upper_bounds)
    )
    background = sol.x[-1]
    residual_rms = np.sqrt(np.mean(sol.fun ** 2))
    success = sol.success

    # Extract fit results for each particle in the cluster.
    results = []
    for j, i in enumerate(cluster):
        k = j * 4
        amplitude, x, y, sigma = sol.x[k:k + 4]
        results.append(
            FitResult(
                i=i,
                x=float(x + x_min),
                y=float(y + y_min),
                amplitude=float(amplitude),
                sigma=float(sigma),
                background=float(background),
                residual_rms=float(residual_rms),
                success=success
            )
        )

    return results
        

def _group_clusters(
        coords: np.ndarray,
        threshold: float
) -> List[List[int]]:

    """Group particle indices whose centres are within `threshold` pixels of each other using a KD-tree.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates.
    threshold : float
        Distance threshold for grouping.

    Returns
    -------
    clusters : List[List[int]]
        List of clusters, where each cluster is a list of indices of `coords`.
    """

    if len(coords) == 0:
        clusters = []
        return clusters

    # Use a KD-tree to find pairs of coordinates within a distance threshold.
    tree = KDTree(coords)
    pairs = tree.query_pairs(threshold)
    parent = list(range(len(coords)))

    # Use a union-find structure to group pairs into clusters.
    for i, j in pairs:
        parent, i_root = _find_root(parent, i)
        parent, j_root = _find_root(parent, j)
        if i_root != j_root:
            parent[j_root] = i_root

    # Collect indices of each cluster.
    clusters = defaultdict(list)
    for i in range(len(coords)):
        _, i_root = _find_root(parent, i)
        clusters[i_root].append(i)
    clusters = list(clusters.values())

    return clusters


def _find_root(
        parent: List[int],
        i: int
) -> Tuple[List[int], int]:

    """Find the root of `i` in a union-find structure.

    Parameters
    ----------
    parent : List[int]
        Parent list in the union-find structure.
    i : int
        Index for which to find the root.

    Returns
    -------
    parent : List[int]
        Parent list with path compression applied.
    i : int
        Root index.
    """

    # Apply path compression while finding the root.
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]

    return parent, i


class GaussianMixtureObjective:

    """Objective function for fitting a Gaussian mixture model to a cluster of particles.

    Parameters
    ----------
    n_gaussians : int
        Number of Gaussians in the mixture.
    x : np.ndarray
        x values at which to evaluate the Gaussian mixture.
    y : np.ndarray
        y values at which to evaluate the Gaussian mixture.
    data : np.ndarray
        Data to fit.
    """

    def __init__(
            self,
            n_gaussians: int,
            x: np.ndarray,
            y: np.ndarray,
            data: np.ndarray
    ) -> None:
        
        self.n_gaussians = n_gaussians
        self.x = x
        self.y = y
        self.data = data
        self._last_params = None
        self._cache = None

    def compute_residuals(self, params: np.ndarray) -> np.ndarray:

        """Compute residuals between Gaussian mixture and data.
        
        Parameters
        ----------
        params : np.ndarray
            Parameters of the Gaussian mixture.
        
        Returns
        -------
        residuals : np.ndarray
            Residuals between Gaussian mixture and data.
        """

        cache = self._compute_cache(params)
        residuals = (cache['gmm'] - self.data).ravel()

        return residuals

    def compute_jacobian(self, params: np.ndarray) -> np.ndarray:

        """Compute Jacobian of Gaussian mixture residuals.

        Parameters
        ----------
        params : np.ndarray
            Parameters of the Gaussian mixture.
        
        Returns
        -------
        jac : np.ndarray
            Jacobian of Gaussian mixture residuals.
        """

        cache = self._compute_cache(params)
        dx = cache['dx']
        dy = cache['dy']
        r2 = cache['r2']
        sigma = cache['sigma']
        sigma2 = cache['sigma2']
        exp_term = cache['exp_term']
        g = cache['g']

        n_pixels = self.x.size
        n_params = len(params)
        jac = np.empty((n_pixels, n_params), dtype=np.float64)
        jac[:, 0:-1:4] = exp_term.reshape(self.n_gaussians, n_pixels).T
        jac[:, 1:-1:4] = (g * dx / sigma2).reshape(self.n_gaussians, n_pixels).T
        jac[:, 2:-1:4] = (g * dy / sigma2).reshape(self.n_gaussians, n_pixels).T
        jac[:, 3:-1:4] = (g * r2 / (sigma * sigma2)).reshape(self.n_gaussians, n_pixels).T
        jac[:, -1] = 1.0

        return jac

    def _compute_cache(self, params: np.ndarray) -> Dict[str, np.ndarray]:

        """Compute intermediate values for the Gaussian mixture.
        
        Parameters
        ----------
        params : np.ndarray
            Parameters of the Gaussian mixture.
        
        Returns
        -------
        cache : Dict[str, np.ndarray]
            Dictionary containing intermediate values for the Gaussian mixture.
        """

        if (self._last_params is not None) and (self._cache is not None) and np.array_equal(params, self._last_params):
            cache = self._cache
            return cache

        p = params[:-1].reshape(self.n_gaussians, 4, 1, 1)
        amplitude, x0, y0, sigma = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        background = params[-1]

        dx = self.x - x0
        dy = self.y - y0
        r2 = dx ** 2 + dy ** 2
        sigma2 = sigma ** 2

        exp_term = np.exp(-r2 / (2 * sigma2))
        g = amplitude * exp_term
        gmm = g.sum(axis=0) + background

        cache = {
            'dx': dx,
            'dy': dy,
            'r2': r2,
            'sigma': sigma,
            'sigma2': sigma2,
            'exp_term': exp_term,
            'g': g,
            'gmm': gmm
        }

        self._last_params = params.copy()
        self._cache = cache

        return cache
