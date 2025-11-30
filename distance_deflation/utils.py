from scipy.sparse.linalg import svds
import numpy as np

def get_orthogonal_vec(dim, num_vec, which = "LM") -> np.ndarray:
    """
    Generate an orthonormal set of vectors in R^dim using SVD on a random matrix.

    Parameters
    ----------
    dim : int
        Dimension of the ambient space.
    num_vec : int
        Number of orthonormal vectors to return. Must satisfy 1 <= num_vec < dim.
        which : {"LM", "SM"}, optional (default="LM")
        Which singular values to find:
        - "LM": largest magnitude
        - "SM": smallest magnitude
        (Directly passed to `svds`.)
        
    Returns
    -------
    U : np.ndarray of shape (dim, num_vec)
        Matrix whose columns form an approximately orthonormal set of vectors.
    """
    
    if num_vec >= dim:
            raise ValueError("num_vec must be less than dim for svds to work.")
    
    # Random matrix; its left singular vectors form an orthonormal basis.
    random_matrix = np.random.rand(dim, dim)
    U, _, _ = svds(random_matrix, k=num_vec, which=which)
    return U

def symmetric_gaussian(D: np.ndarray, bd: float) -> np.ndarray:
    """
    Compute a symmetrically normalized Gaussian kernel from a squared distance matrix.

    Given squared distances D_ij and a bandwidth bd, this function computes

        K_ij = exp(-D_ij / bd)

    and then applies symmetric normalization:

        K_sym = D^{-1/2} K D^{-1/2},

    where D is the diagonal degree matrix with D_ii = sum_j K_ij.

    Parameters
    ----------
    D : np.ndarray of shape (n, n)
        Pairwise distance (or squared distance) matrix.
    bd : float
        Bandwidth parameter in the Gaussian kernel.

    Returns
    -------
    K_sym : np.ndarray of shape (n, n)
        Symmetrically normalized Gaussian kernel matrix.
    """
    
    K = np.exp(-D / bd)

    # Degree vector: row sums of K
    row_sums = np.sum(K, axis=1)
    if np.any(row_sums == 0):
        raise ValueError("At least one row of K has zero sum; cannot normalize.")

    # D^{-1/2} as a diagonal matrix
    d = np.diag(1.0 / np.sqrt(row_sums))

    K_sym = d @ K @ d

    return K_sym

def rs_gaussian(D: np.ndarray, bd: float) -> np.ndarray:
    """
    Compute a row-stochastic Gaussian kernel from a distance matrix.

    Given distances D_ij and bandwidth bd, this function computes

        K_ij = exp(-D_ij / bd)

    and then normalizes each row to sum to 1:

        K_rs[i, :] = K[i, :] / sum_j K[i, j].

    Parameters
    ----------
    D : np.ndarray of shape (n, n)
        Pairwise distance (or squared distance) matrix.
    bd : float
        Bandwidth parameter in the Gaussian kernel.

    Returns
    -------
    K_rs : np.ndarray of shape (n, n)
        Row-stochastic Gaussian kernel matrix.
    """
    K = np.exp(-D / bd)
    row_sums = np.sum(K, axis=1, keepdims=True)

    if np.any(row_sums == 0):
        raise ValueError("At least one row of K has zero sum; cannot normalize.")

    K_rs = K / row_sums
    return K_rs

def neighborhood_dist(dist: np.ndarray, k: int) -> np.ndarray:
    """
    Compute a k-nearest-neighbor indicator based on a distance matrix.

    For each row i, this function finds the k-th smallest distance value
    (using `np.partition`) and marks all entries <= that threshold as neighbors.


    Parameters
    ----------
    dist : np.ndarray of shape (n, n)
        Pairwise distance matrix.
    k : int
        Number of neighbors.

    Returns
    -------
    neighbor_indicator : np.ndarray of shape (n, n), dtype=bool
        Boolean matrix where [i, j] is True if j is among the first (k)
        neighbors of i in terms of smallest distance.
    """
    # k-th smallest value in each row (0-based index)
    kth = np.partition(dist, k, axis=1)[:, k].reshape(-1, 1)

    # Mark as neighbor if distance <= k-th smallest distance
    neighbor_indicator = dist <= kth

    return neighbor_indicator


def neighborhood_dist_overlap(
    true_dist: np.ndarray,
    dist1: np.ndarray,
    dist2: np.ndarray,
    k: int
):
    """
    Compare k-NN neighborhood overlap between an oracle distance and two candidates.

    For each distance matrix, we:
      1. Compute the k-nearest-neighbor indicator (using `neighborhood_dist`).
      2. Compare the candidate neighborhood to the 'true' neighborhood row-wise.
      3. Compute the average overlap rate over all rows.

    Overlap for a single row i is:
        overlap_rate_i = |N_true(i) ∩ N_dist(i)| / |N_true(i)|,
    and the final value is the mean of these overlap rates across i.

    Parameters
    ----------
    true_dist : np.ndarray of shape (n, n)
        Ground-truth pairwise distance matrix.
    dist1 : np.ndarray of shape (n, n)
        First candidate distance matrix.
    dist2 : np.ndarray of shape (n, n)
        Second candidate distance matrix.
    k : int
        Number of neighbors.

    Returns
    -------
    dist1_rate : float
        Average neighborhood overlap rate between true_dist and dist1.
    dist2_rate : float
        Average neighborhood overlap rate between true_dist and dist2.
    """
    # k-NN indicators for each distance matrix
    true_indicator = neighborhood_dist(true_dist, k)
    dist1_indicator = neighborhood_dist(dist1, k)
    dist2_indicator = neighborhood_dist(dist2, k)

    # Size of the true neighborhood per row (column vector)
    true_neighbor_size = np.sum(true_indicator, axis=1, keepdims=True)

    # For numerical safety, ensure there is at least one true neighbor per row
    if np.any(true_neighbor_size == 0):
        raise ValueError("At least one row has zero true neighbors; cannot compute overlap.")

    # Overlap counts for dist1
    dist1_overlap = true_indicator & dist1_indicator
    dist1_rate = np.mean(
        np.sum(dist1_overlap, axis=1, keepdims=True) / true_neighbor_size
    )

    # Overlap counts for dist2
    dist2_overlap = true_indicator & dist2_indicator
    dist2_rate = np.mean(
        np.sum(dist2_overlap, axis=1, keepdims=True) / true_neighbor_size
    )

    return dist1_rate, dist2_rate