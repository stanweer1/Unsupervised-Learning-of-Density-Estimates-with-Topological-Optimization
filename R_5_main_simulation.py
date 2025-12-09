import numpy as np
import torch
import pandas as pd
import time
import sys
import os
import random

# Import all our custom modules
import R_1_data_generator as data_gen
import R_2_tda_optimizer as tda
import R_3_kde_metrics as metrics
import R_4_visualize as viz

# ===================================================================
# SEEDING FUNCTION
# ===================================================================
def set_seeds(seed):
    """
    Sets all random seeds for complete reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"  All seeds set to {seed}. Using deterministic algorithms.")

# ===================================================================
# SIMULATION CONFIG
# ===================================================================
N_ITERS_TDA = 250 # Iterations for TDA optimization

# Define the full benchmark suite
BENCHMARK_SUITE = {
    # --- 1D Datasets ---
    '1D_bimodal': {
        'data_func': lambda n: data_gen.generate_samples_1D(data_gen.bimodal_gaussian_pdf, n_samples=n),
        'pdf_func': data_gen.bimodal_gaussian_pdf,
        'dims': 1, 'n_samples': 5000, 'grid_per_dim': 200
    },
    '1D_complex': {
        'data_func': lambda n: data_gen.generate_samples_1D(data_gen.complex_mixture_pdf, n_samples=n),
        'pdf_func': data_gen.complex_mixture_pdf,
        'dims': 1, 'n_samples': 5000, 'grid_per_dim': 200
    },
    # --- 2D Datasets ---
    '2D_clusters': {
        'data_func': lambda n: data_gen.generate_samples_2D(data_gen.clusters_pdf, n_samples=n),
        'pdf_func': data_gen.clusters_pdf,
        'dims': 2, 'n_samples': 2000, 'grid_per_dim': 100
    },
    '2D_elliptical': {
        'data_func': lambda n: data_gen.generate_samples_2D(data_gen.elliptical_pdf, n_samples=n),
        'pdf_func': data_gen.elliptical_pdf,
        'dims': 2, 'n_samples': 2000, 'grid_per_dim': 100
    },
    '2D_weibull': {
        'data_func': lambda n: data_gen.generate_samples_2D(data_gen.weibull_pdf, n_samples=n),
        'pdf_func': data_gen.weibull_pdf,
        'dims': 2, 'n_samples': 2000, 'grid_per_dim': 100
    },
    'MNIST': {
        'data_func': lambda n: data_gen.load_mnist_pointcloud(digit=n),
        'pdf_grid_func': lambda n: data_gen.load_mnist_true_density(digit=n),
        'dims': 2,
        'n_samples': 100,
        'grid_per_dim': 100
    },    
    # --- 3D Datasets ---
    '3D_gauss': {
        'data_func': lambda n: data_gen.generate_samples_3D(data_gen.gauss_pdf_3, n_samples=n),
        'pdf_func': data_gen.gauss_pdf_3,
        'dims': 3, 'n_samples': 1000, 'grid_per_dim': 70
    },
    '3D_heavytail': {
        'data_func': lambda n: data_gen.generate_samples_3D(data_gen.heavytail_spike_3, n_samples=n),
        'pdf_func': data_gen.heavytail_spike_3,
        'dims': 3, 'n_samples': 1000, 'grid_per_dim': 70
    },
    '3D_manifold': {
        'data_func': lambda n: data_gen.generate_samples_3D(data_gen.manifold, n_samples=n),
        'pdf_func': data_gen.manifold,
        'dims': 3, 'n_samples': 1000, 'grid_per_dim': 70
    },
    # --- 4D Datasets ---
    '4D_gauss': {
        'data_func': lambda n: data_gen.generate_samples_4D(data_gen.gauss_pdf_4, n_samples=n),
        'pdf_func': data_gen.gauss_pdf_4,
        'dims': 4, 'n_samples': 1000, 'grid_per_dim': 40
    },
    '4D_heavytail': {
        'data_func': lambda n: data_gen.generate_samples_4D(data_gen.heavytail_spike_4, n_samples=n),
        'pdf_func': data_gen.heavytail_spike_4,
        'dims': 4, 'n_samples': 1000, 'grid_per_dim': 40
    },
    '4D_heavymixture': {
        'data_func': lambda n: data_gen.generate_samples_4D(data_gen.heavy_mixture_4, n_samples=n),
        'pdf_func': data_gen.heavy_mixture_4,
        'dims': 4, 'n_samples': 1000, 'grid_per_dim': 40
    }
}

# ===================================================================
# SINGLE JOB FUNCTION
# ===================================================================

def run_single_job(dataset_name, seed):
    """
    Runs a single simulation for one dataset and one seed.
    """
    print(f"\n--- Running Job: Dataset={dataset_name}, Seed={seed} ---")
    
    # 1. Set Seed
    set_seeds(seed)
    
    # 2. Get Config
    config = BENCHMARK_SUITE.get(dataset_name)
    if config is None:
        print(f"Error: Dataset '{dataset_name}' not found in BENCHMARK_SUITE.")
        return

    n_samples = config['n_samples']
    grid_per_dim = config['grid_per_dim']
    print(f"Running dataset: {dataset_name} (D={config['dims']}, N={n_samples}, Grid={grid_per_dim}^{config['dims']})")
    
    # 3. Generate Data
    if dataset_name == 'MNIST':
        data_nd = config['data_func'](seed % 10)
    else:
        data_nd = config['data_func'](n_samples)
    data_torch = torch.tensor(data_nd, dtype=torch.float32)

    # 4. Get all bandwidths
    bandwidths, times = metrics.get_kde_bandwidths(data_nd, verbose=True)
    
    # TDA method
    print("  Optimizing TDA bandwidth...")
    t = time.perf_counter()
    bw_tda, losses, density, shape, bounds = tda.optimize_bandwidth_nd(
        data_torch, 
        resolution_per_dim=grid_per_dim, 
        n_iters=N_ITERS_TDA,
        lr=0.01,
        alpha_pe=1.0,
        alpha_tp=1.0,
        lambda_smooth=0.0,
        only_H0=True,
        verbose=False
    )
    times['TDA'] = time.perf_counter() - t
    bandwidths['TDA'] = bw_tda
    print(f"    'TDA' complete. (bw={bw_tda:.4f})")

    # 5. Create Grid for Metrics
    if dataset_name == 'MNIST':
        # === MNIST MODE ===
        Z_true_norm, linspaces = config["pdf_grid_func"](seed % 10)
        Z_true_norm = Z_true_norm.T

        xs, ys = linspaces
        Xg, Yg = np.meshgrid(xs, ys, indexing='xy')
        grid_points = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
        grid_shape = (len(xs), len(ys))
        grid_bounds = None
        grid_coords = None   # not required
        grid_points_np = grid_points.astype(np.float64)

    else:
        # === ANALYTIC MODE ===
        grid_points, grid_shape, grid_bounds = tda.create_grid_nd(
            data_torch, grid_per_dim
        )
        grid_points_np = grid_points.cpu().numpy()
        linspaces = [np.linspace(b[0], b[1], grid_per_dim) for b in grid_bounds]
        grid_coords = np.meshgrid(*linspaces, indexing='ij')
        Z_true = config['pdf_func'](*grid_coords)
        Z_true_norm = metrics.normalize_pdf_nd(Z_true, linspaces)
        grid_points_np = grid_points.cpu().numpy()

    # 6. Evaluate all KDEs on the grid
    Z_kdes = metrics.evaluate_all_kdes(data_nd, bandwidths, grid_points_np, grid_shape, linspaces)

    # 8. Compute Metrics and Store Results
    print("  Computing final metrics...")
    all_results = []
    for method, Z_kde_norm in Z_kdes.items():
        if dataset_name == 'MNIST' and type(Z_kde_norm) is not int:
            Z_kdes[method] = Z_kde_norm.T
            Z_kde_norm = Z_kdes[method]
            true_mass = np.sum(Z_true_norm)
            kde_mass = np.sum(Z_kde_norm)
            if kde_mass > 0:
                Z_kde_norm = Z_kde_norm * (true_mass / kde_mass)
        kld = metrics.kl_divergence_nd(Z_true_norm, Z_kde_norm, linspaces)
        emd = metrics.emd_nd(Z_true_norm, Z_kde_norm, linspaces)
        
        print(f"    {method:<12} KLD: {kld:.6f}, EMD: {emd:.6f}")
        
        all_results.append({
            'seed': seed,
            'dataset': dataset_name,
            'method': method,
            'bandwidth': bandwidths[method],
            'kld': kld,
            'emd': emd,
            'time': times[method]
        })

    # 9. Save this job's results to a unique CSV
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{dataset_name}_seed_{seed}.csv")
    
    df = pd.DataFrame(all_results)
    df.to_csv(filename, index=False)
    
    print(f"\n--- Job Complete ---")
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 5_run_job.py <dataset_name> <seed>")
        print("\nAvailable datasets:")
        for name in BENCHMARK_SUITE.keys():
            print(f"  {name}")
        sys.exit(1)

    dataset_name = sys.argv[1]
    
    try:
        seed = int(sys.argv[2])
    except ValueError:
        print("Error: Seed must be an integer.")
        sys.exit(1)
        
    run_single_job(dataset_name, seed)