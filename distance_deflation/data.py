import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import linear_sum_assignment

class Deflation:
    """
    Distance Deflation Algorithm: https://www.arxiv.org/pdf/2507.18520.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features). Each row is a data point.
    verbose : bool, optional (default=True)
        Whether to print summary statistics in `get_S_distance`.
    plot : bool, optional (default=False)
        Whether to plot a histogram in `get_S_distance`.
    n_jobs : int, optional (default=4)
        Number of jobs for `pairwise_distances` (passed directly to sklearn).
    """
    def __init__(self, X, verbose = True, plot = False, n_jobs = 4): 
        self.X = np.asarray(X)
        
        # Compute pairwise squared distances and store in self.S_distance
        self.get_S_distance(verbose = verbose, plot = plot, n_jobs = n_jobs)
        
        # Will be filled by debias_LA()
        self.squared_noise_mag = None        # (n, 1) array of row-wise noise estimates.
        self.permute_list = None             # assignment results 
        self.b_list = None                   # assignment costs
        self.denoised = None                 # (n, n) array of debiased distances.
        
        
    def get_S_distance(self, plot = False, verbose = False, n_jobs = 4): 
        """
        Compute the squared Euclidean pairwise distance matrix for self.X.

        The result is stored in self.S_distance, with the diagonal set to +inf
        to avoid self-loops in downstream matching.

        Parameters
        ----------
        plot : bool, optional (default=False)
            If True, plot a histogram of all pairwise squared distances.
        verbose : bool, optional (default=False)
            If True, print basic summary statistics of the distances.
        n_jobs : int, optional (default=4)
            Number of parallel jobs passed to sklearn.metrics.pairwise_distances.
        """
        
        # Compute full n x n matrix of squared Euclidean distances
        self.S_distance = pairwise_distances(
            self.X, self.X, metric="sqeuclidean", n_jobs=n_jobs
        )

        # Plot histogram of distances if requested
        if plot:
            plt.hist(self.S_distance.flatten(), bins=200)
            plt.ylabel("Count")
            plt.xlabel("Squared distance")
            plt.title("Histogram of Squared Pairwise Euclidean Distances")
            plt.show()

        # Exclude zeros (diagonal) for summary stats
        non_zero_distances = self.S_distance[self.S_distance != 0].flatten()

        if verbose and non_zero_distances.size > 0:
            print(f"Min Distance:           {np.min(non_zero_distances):.2e}")
            print(f"5th Percentile:         {np.quantile(non_zero_distances, 0.05):.2e}")
            print(f"10th Percentile:        {np.quantile(non_zero_distances, 0.10):.2e}")
            print(f"25th Percentile:        {np.quantile(non_zero_distances, 0.25):.2e}")
            print(f"Median Distance:        {np.median(non_zero_distances):.2e}")
            print(f"75th Percentile:        {np.quantile(non_zero_distances, 0.75):.2e}")
            print(f"Max Distance:           {np.max(non_zero_distances):.2e}")

        # Set diagonal to infinity to avoid self-assignments in linear_sum_assignment
        np.fill_diagonal(self.S_distance, np.inf)

    def _LA(self, S_distance):
        """
        Perform one step of linear assignment on a distance matrix.

        This finds an optimal one-to-one matching (permutation) that minimizes
        the sum of the selected distances, then:
          - records the matched costs b,
          - sets those matched entries in S_distance to +inf so that they
            won't be selected in subsequent calls.

        Parameters
        ----------
        S_distance : np.ndarray
            Current working distance matrix (n x n). Modified in-place.

        Returns
        -------
        perm : np.ndarray of shape (n,)
            Permutation such that row i is matched to column perm[i].
        S_distance : np.ndarray
            The same matrix as input, but with matched entries set to +inf.
        b : np.ndarray of shape (n, 1)
            Column vector of matched costs (squared distances) for each row.
        """
        
        n = S_distance.shape[0]
        if S_distance.shape[0] != S_distance.shape[1]:
            raise ValueError("S_distance must be a square matrix.")

        row_ind, col_ind = linear_sum_assignment(S_distance)
        
        # Extract costs for the assignment
        b = S_distance[row_ind,col_ind].reshape(-1,1).copy()

         # Set matched entries to infinity so they won't be reused
        S_distance[row_ind, col_ind] = np.inf
        return col_ind, S_distance, b

    def debias_LA(self):
        '''
        Implementation of Algorithm 1 in https://www.arxiv.org/pdf/2507.18520.
        '''
        
        S_distance = self.S_distance.copy()

        permute_list = []
        b_list = []
        
        for _ in range(2):
            permute, S_distance, b = self._LA(S_distance)
            permute_list.append(permute)
            b_list.append(b)
            
        b_trio = self.S_distance[permute_list[0],permute_list[1]].reshape(-1,1)
        b_list.append(b_trio)
        
        # Estimate per-observation noise magnitude
        noise_mage_est = 1/2*(b_list[0] + b_list[1] - b_trio)
        
        self.permute_list = permute_list
        self.b_list = b_list
        
        self.squared_noise_mag = noise_mage_est
        self.denoised = self.S_distance.copy() - self.squared_noise_mag - self.squared_noise_mag.T

    
                                     



