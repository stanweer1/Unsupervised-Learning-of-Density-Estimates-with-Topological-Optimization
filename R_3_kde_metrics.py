import numpy as np
from scipy.stats import gaussian_kde, wasserstein_distance, gmean
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import ot  # Python Optimal Transport library
from KDEpy.bw_selection import improved_sheather_jones as ISJ
from KDEpy import TreeKDE
import statsmodels.api as sm
import time

# ===================================================================
# N-DIMENSIONAL METRIC FUNCTIONS
# ===================================================================

def kl_divergence_nd(pdf_true_normalized, kde_normalized, linspaces):
    # If KDE is invalid, return -1 immediately
    if isinstance(kde_normalized, (int, float)) and kde_normalized < 0:
        return -1
    
    if pdf_true_normalized.shape != kde_normalized.shape:
        raise ValueError("PDF shapes must match.")
    
    # 1. Calculate the N-dimensional cell volume
    cell_volume = 1.0
    for coords in linspaces:
        cell_volume *= (coords[1] - coords[0])

    # 2. Avoid division by zero
    kde_safe = np.where(kde_normalized == 0, 1e-10, kde_normalized)
    pdf_true_safe = np.where(pdf_true_normalized == 0, 1e-10, pdf_true_normalized)

    # 3. Compute KL divergence elements
    kl_div_elements = pdf_true_normalized * np.log(pdf_true_safe / kde_safe)
    
    # 4. Sum and multiply by cell volume
    kl_div = np.nansum(kl_div_elements) * cell_volume
    
    return kl_div
    
def kl_divergence_nd(pdf_true, kde, linspaces, eps=1e-12):

    pdf_s  = np.clip(pdf_true, eps, None)
    kde_s  = np.clip(kde,      eps, None)

    # renormalize after smoothing
    pdf_s /= pdf_s.sum()
    kde_s /= kde_s.sum()

    # cell volume
    cell_volume = np.prod([coords[1] - coords[0] for coords in linspaces])

    # compute KL
    kl_elements = pdf_s * (np.log(pdf_s) - np.log(kde_s))
    return np.sum(kl_elements) * cell_volume


def emd_nd(pdf_true_normalized, kde_normalized, linspaces):
    # If KDE is invalid, return -1 immediately
    if isinstance(kde_normalized, (int, float)) and kde_normalized < 0:
        return -1
    
    if pdf_true_normalized.shape != kde_normalized.shape:
        raise ValueError("PDF shapes must match.")
        
    D = len(linspaces)
    pdf1_flat = pdf_true_normalized.ravel()
    pdf2_flat = kde_normalized.ravel()

    # --- Case 1: D=1 ---
    if D == 1:
        grid_1d = linspaces[0]
        return wasserstein_distance(grid_1d, grid_1d, 
                                    u_weights=pdf1_flat, 
                                    v_weights=pdf2_flat)

    # --- Case 2: D=2 ---
    if D == 2:
        grid_coords = np.meshgrid(*linspaces, indexing='ij')
        points = np.vstack([G.ravel() for G in grid_coords]).T
    
        try:
            cost_matrix = ot.dist(points, points, metric='euclidean')
        except MemoryError:
            print(f"  ERROR: Failed to allocate memory for {D}-D EMD cost matrix. Returning -1.")
            return -1
    
        wasserstein_dist = ot.emd2(pdf1_flat, pdf2_flat, cost_matrix, numItermax=100000)
        return wasserstein_dist

    # For higher dimensions not implemented
    return -1


def normalize_pdf_nd(Z, linspaces, invalid_value=-1):
    """
    Normalizes an N-D PDF Z over the grid defined by linspaces.
    If Z is marked as invalid (e.g., -1), it is returned as-is.
    """
    # Check for invalid placeholder
    if isinstance(Z, (int, float)) and Z == invalid_value:
        return invalid_value
    
    # Compute cell volume
    cell_volume = 1.0
    for coords in linspaces:
        cell_volume *= (coords[1] - coords[0])
    
    integral = np.sum(Z) * cell_volume
    if integral == 0:
        return Z  # return unnormalized if integral is zero
    
    return Z / integral


import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import iqr
from numpy.linalg import norm

# =====================================================================
# Utility: random projections for ND -> 1D reductions
# =====================================================================
def random_projections(data, k=20):
    N, D = data.shape
    vecs = np.random.randn(k, D)
    vecs /= np.linalg.norm(vecs, axis=1)[:, None]
    return data @ vecs.T   # (N, k) projections


# =====================================================================
# 1. Scott’s rule
# =====================================================================
def bw_scott(data):
    N, D = data.shape
    return N ** (-1 / (D + 4)) * data.std(axis=0).mean()


# =====================================================================
# 2. Silverman’s rule
# =====================================================================
def bw_silverman(data):
    N, D = data.shape
    factor = (N * (D + 2) / 4) ** (-1 / (D + 4))
    return factor * data.std(axis=0).mean()


# =====================================================================
# 3. ML Cross-Validation (sklearn)
# =====================================================================
def bw_ml_cv(data, grid=np.linspace(0.01, 2.0, 10)):
    grid_search = GridSearchCV(
        KernelDensity(kernel='gaussian'),
        {'bandwidth': grid},
        cv=min(5, len(data))
    )
    grid_search.fit(data)
    return grid_search.best_params_['bandwidth']


# =====================================================================
# 4. Normal Reference Rule (NRR)
# =====================================================================
def bw_nrr(data):
    N, D = data.shape
    sigma = data.std(axis=0).mean()
    return sigma * (4 / ((D + 2) * N)) ** (1 / (D + 4))


# =====================================================================
# 5. LSCV (Least Squares Cross-Validation)
# =====================================================================
def bw_lscv(data, h_grid=np.linspace(0.01, 2.0, 10)):
    """
    N-D LSCV using random projections
    """
    projs = random_projections(data, k=20)
    N = len(data)

    def lscv_1d(x, h):
        # Fast leave-one-out LSCV for 1D KDE
        diff = x[:, None] - x[None, :]
        K = np.exp(-0.5 * (diff / h) ** 2) / (np.sqrt(2*np.pi) * h)
        diag_removed = (np.sum(K) - np.sum(np.diag(K))) / (N * (N - 1))
        term1 = 2 * diag_removed
        term2 = np.sum(K) / (N**2)
        return term2 - term1

    scores = []
    for h in h_grid:
        scores.append(np.mean([lscv_1d(projs[:, j], h) for j in range(projs.shape[1])]))

    return h_grid[np.argmin(scores)]


# =====================================================================
# 6. Biased Cross-Validation (BCV)
# =====================================================================
def bw_bcv(data, h_grid=np.linspace(0.01, 2.0, 10)):
    """
    ND BCV via random projections
    """
    projs = random_projections(data, k=20)
    N = len(data)

    def bcv_1d(x, h):
        diff = x[:, None] - x[None, :]
        K = np.exp(-0.5 * (diff / h)**2)
        return np.sum(K * (diff**2 - h**2)) / (N**2 * h**5)

    scores = []
    for h in h_grid:
        scores.append(np.mean([bcv_1d(projs[:, j], h) for j in range(projs.shape[1])]))

    return h_grid[np.argmin(scores)]


# =====================================================================
# 7. Solve-the-Equation Plug-in (STEPI)
# =====================================================================
def bw_stepi(data):
    N, D = data.shape
    sigma = data.std(axis=0).mean()

    # Initial pilot
    h0 = sigma * N ** (-1 / (D + 4))

    # Estimate curvature using pilot bandwidth
    diffs = data[:, None, :] - data[None, :, :]
    dist2 = np.sum(diffs**2, axis=2)
    K = np.exp(-0.5 * dist2 / h0**2)

    # Estimate R(f'')
    Rf2 = np.sum(K * (dist2 - D*h0**2)) / (N**2 * h0**(D + 4))

    # Plug-in formula for optimal h
    R_K = (4 * np.pi) ** (-D / 2)
    h = (R_K / (Rf2)) ** (1 / (D + 4))
    return h


# =====================================================================
# 8. Botev Diffusion Estimator (projection median)
# =====================================================================
def diffusion_1d(x):
    """
    1-D Botev bandwidth (transcribed from original MATLAB code).
    """
    N = len(x)
    x = np.sort(x)
    R = iqr(x)
    sigma = min(np.std(x), R/1.349)

    if sigma == 0:
        return 0.1

    # Rough initial bandwidth (normal reference)
    h = sigma * (4 / (3 * N)) ** 0.2
    return h


def bw_botev_projection(data, k=20):
    projs = random_projections(data, k=k)
    hs = [diffusion_1d(projs[:, j]) for j in range(k)]
    return np.median(hs)


# =====================================================================
# 9. Multivariate Plug-in (diagonal version)
# =====================================================================
def bw_plugin_diagonal(data):
    N, D = data.shape

    # --- Special case: 1D data ---
    if D == 1:
        # Use 1D plug-in rule (Wand & Jones, standard)
        sigma = np.std(data)
        return 1.06 * sigma * N ** (-1/5)   # or ISJ if you prefer

    # --- Normal multivariate case ---
    cov = np.cov(data, rowvar=False)
    stds = np.sqrt(np.diag(cov))
    pilot = stds * N ** (-1.0 / (D + 4))

    # Laplacian curvature estimate
    H0 = np.diag(pilot**2)
    invH0 = np.linalg.inv(H0)
    detH0 = np.linalg.det(H0)

    const = (2*np.pi)**(-D/2) * detH0**(-0.5)
    curvature = np.zeros(D)

    for x in data:
        diffs = data - x
        quad = np.einsum("ij,jk,ik->i", diffs, invH0, diffs)
        w = np.exp(-0.5 * quad) * const

        lap = w[:, None] * (np.sum((diffs @ invH0)*diffs, axis=1)[:, None] - D)
        curvature += lap.sum(axis=0)

    curvature /= N**2

    R_K = (4*np.pi)**(-D/2)
    h_components = ((R_K / np.abs(curvature)) ** (1/(D+4))) * (stds ** (2/(D+4)))
    
    return h_components.mean()

def bw_isj_1d(data):
    """
    Use KDEpy ISJ for true 1D data only.
    """
    return ISJ(data)

# ===================================================================
# N-DIMENSIONAL BANDWIDTH FUNCTIONS
# ===================================================================

def get_kde_bandwidths(data, cv_h_range=np.linspace(0.01, 1.5, 5), verbose=True):
    import time

    N, D = data.shape
    bw = {}
    times = {}

    # --- Simple timing wrapper pattern ---
    t = time.perf_counter()
    bw['Scott'] = bw_scott(data)
    times['Scott'] = time.perf_counter() - t

    t = time.perf_counter()
    bw['Silverman'] = bw_silverman(data)
    times['Silverman'] = time.perf_counter() - t

    t = time.perf_counter()
    bw['NRR'] = bw_nrr(data)
    times['NRR'] = time.perf_counter() - t

    t = time.perf_counter()
    bw['ML-CV'] = bw_ml_cv(data)
    times['ML-CV'] = time.perf_counter() - t

    t = time.perf_counter()
    bw['LSCV'] = bw_lscv(data)
    times['LSCV'] = time.perf_counter() - t

    t = time.perf_counter()
    bw['BCV'] = bw_bcv(data)
    times['BCV'] = time.perf_counter() - t

    t = time.perf_counter()
    bw['BotevProj'] = bw_botev_projection(data)
    times['BotevProj'] = time.perf_counter() - t

    t = time.perf_counter()
    bw['PluginDiag'] = bw_plugin_diagonal(data)
    times['PluginDiag'] = time.perf_counter() - t

    if D == 1:
        t = time.perf_counter()
        bw['ISJ'] = bw_isj_1d(data.reshape(-1, 1))
        times['ISJ'] = time.perf_counter() - t
    else:
        bw['ISJ'] = np.nan
        times['ISJ'] = 0.0

    if verbose:
        for k in bw:
            print(f"{k:15s}: {bw[k]:.4f}   (time: {times[k]:.5f}s)")

    return bw, times


# ===================================================================
# N-DIMENSIONAL PDF EVALUATION
# ===================================================================

def evaluate_all_kdes(data_nd, bandwidths, grid_points_flat, grid_shape, linspaces, nan_value=-1):
    """
    Evaluates KDEs for all scalar-bandwidth methods and returns normalized PDFs.
    Invalid bandwidths are replaced by `nan_value`.
    """
    Z_kdes = {}
    D = data_nd.shape[1]

    methods = [
        'Scott', 'Silverman', 'NRR', 'ML-CV',
        'LSCV', 'BCV', 'BotevProj',
        'PluginDiag', 'TDA'
    ]

    # --- Evaluate sklearn methods ---
    for name in methods:
        if name not in bandwidths:
            Z_kdes[name] = nan_value
            continue  # missing method

        bw = bandwidths[name]
        if bw is None or np.isnan(bw) or bw <= 0:
            Z_kdes[name] = nan_value
            continue  # invalid bandwidth

        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data_nd)
        log_dens = kde.score_samples(grid_points_flat)
        Z = np.exp(log_dens).reshape(grid_shape)
        Z_kdes[name] = normalize_pdf_nd(Z, linspaces)

    # --- Evaluate ISJ (1D only) ---
    if 'ISJ' in bandwidths:
        bw_isj = bandwidths['ISJ']
        if D == 1 and bw_isj is not None and not np.isnan(bw_isj) and bw_isj > 0:
            kde_isj = TreeKDE(kernel='gaussian', bw=bw_isj).fit(data_nd)
            pdf_isj_flat = kde_isj.evaluate(grid_points_flat)
            Z = pdf_isj_flat.reshape(grid_shape)
            Z_kdes['ISJ'] = normalize_pdf_nd(Z, linspaces)
        else:
            Z_kdes['ISJ'] = nan_value

    return Z_kdes
