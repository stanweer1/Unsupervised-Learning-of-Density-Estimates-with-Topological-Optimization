import numpy as np
import torch
import pandas as pd
import sys
import os
import random

# Import custom modules
import R_1_data_generator as data_gen
import R_2_tda_optimizer as tda
import R_3_kde_metrics as metrics

# ===================================================================
# SEEDING FUNCTION
# ===================================================================
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"  All seeds set to {seed}. Using deterministic algorithms.")

# ===================================================================
# SIMULATION CONFIG
# ===================================================================
N_ITERS_TDA = 250  # Iterations for TDA optimization

BENCHMARK_SUITE = {
    '1D_bimodal': {'data_func': lambda n: data_gen.generate_samples_1D(data_gen.bimodal_gaussian_pdf, n_samples=n),
                   'pdf_func': data_gen.bimodal_gaussian_pdf, 'dims': 1, 'n_samples': 5000, 'grid_per_dim': 200},
    '1D_complex': {'data_func': lambda n: data_gen.generate_samples_1D(data_gen.complex_mixture_pdf, n_samples=n),
                   'pdf_func': data_gen.complex_mixture_pdf, 'dims': 1, 'n_samples': 5000, 'grid_per_dim': 200},
    '2D_clusters': {'data_func': lambda n: data_gen.generate_samples_2D(data_gen.clusters_pdf, n_samples=n),
                    'pdf_func': data_gen.clusters_pdf, 'dims': 2, 'n_samples': 2000, 'grid_per_dim': 100},
    '2D_annulus': {'data_func': lambda n: data_gen.generate_samples_2D(data_gen.annulus_pdf, n_samples=n),
                   'pdf_func': data_gen.annulus_pdf, 'dims': 2, 'n_samples': 2000, 'grid_per_dim': 100},
    '3D_heavytail': {'data_func': lambda n: data_gen.generate_samples_3D(data_gen.heavytail_spike_3, n_samples=n),
                     'pdf_func': data_gen.heavytail_spike_3, 'dims': 3, 'n_samples': 1000, 'grid_per_dim': 70},
    '4D_gauss': {'data_func': lambda n: data_gen.generate_samples_4D(data_gen.gauss_pdf_4, n_samples=n),
                 'pdf_func': data_gen.gauss_pdf_4, 'dims': 4, 'n_samples': 1000, 'grid_per_dim': 40}
}

# ===================================================================
# SINGLE JOB FUNCTION
# ===================================================================
def run_single_job(dataset_name, seed, count_on, TP_on, ref_on, only_H0, label="full"):
    print(f"\n--- Running Job: Dataset={dataset_name}, Seed={seed}, Ablation={label} ---")
    
    # 1. Set seed
    set_seeds(seed)
    
    # 2. Load dataset config
    config = BENCHMARK_SUITE.get(dataset_name)
    if config is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in BENCHMARK_SUITE.")
    
    n_samples = config['n_samples']
    grid_per_dim = config['grid_per_dim']
    
    print(f"Dataset: {dataset_name} | D={config['dims']} | N={n_samples} | Grid={grid_per_dim}^{config['dims']}")
    
    # 3. Generate data
    data_nd = config['data_func'](n_samples)
    data_torch = torch.tensor(data_nd, dtype=torch.float32)
    
    # 4. Get bandwidths
    bandwidths, _ = metrics.get_kde_bandwidths(data_nd, verbose=True)
    
    # 5. Optimize TDA bandwidth
    print("  Optimizing TDA bandwidth...")
    bw_tda, losses, density, shape, bounds = tda.optimize_bandwidth_nd(
        data_torch,
        resolution_per_dim=grid_per_dim,
        n_iters=N_ITERS_TDA,
        lr=0.01,
        alpha_pe=count_on,
        alpha_tp=TP_on,
        lambda_smooth=ref_on,
        only_H0=only_H0,
        verbose=False
    )
    bandwidths['TDA'] = bw_tda
    print(f"    'TDA' complete. (bw={bw_tda:.4f})")
    
    # 6. Create grid
    grid_points, grid_shape, grid_bounds = tda.create_grid_nd(data_torch, grid_per_dim)
    grid_points_np = grid_points.cpu().numpy()
    linspaces = [np.linspace(b[0], b[1], grid_per_dim) for b in grid_bounds]
    grid_coords = np.meshgrid(*linspaces, indexing='ij')
    
    # 7. Evaluate KDEs
    print("  Evaluating KDEs on grid...")
    Z_kdes = metrics.evaluate_all_kdes(data_nd, bandwidths, grid_points_np, grid_shape, linspaces)
    
    # 8. True PDF
    Z_true = config['pdf_func'](*grid_coords)
    Z_true_norm = metrics.normalize_pdf_nd(Z_true, linspaces)
    
    # 9. Compute only TDA metrics
    tda_kld, tda_emd = None, None
    for method, Z_kde_norm in Z_kdes.items():
        kld = metrics.kl_divergence_nd(Z_true_norm, Z_kde_norm, linspaces)
        emd = metrics.emd_nd(Z_true_norm, Z_kde_norm, linspaces)
        if method == "TDA":
            tda_kld, tda_emd = kld, emd
        print(f"    {method:<12} KLD: {kld:.6f}, EMD: {emd:.6f}")
    
    if tda_kld is None:
        raise RuntimeError("ERROR: TDA metrics were not computed.")
    
    # 10. Save metrics as NumPy array
    output_dir = "ablation_study"
    os.makedirs(output_dir, exist_ok=True)
    
    np_save_path = os.path.join(
        output_dir,
        f"{dataset_name}_seed_{seed}_{label}.npy"
    )
    np.save(np_save_path, np.array([tda_kld, tda_emd], dtype=float))
    
    print(f"  Saved TDA metrics array to: {np_save_path}")
    print("\n--- Job Complete ---\n")

# ===================================================================
# ENTRY POINT
# ===================================================================
if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python 5_run_job.py <dataset_name> <seed> <count_on> <TP_on> <ref_on> <only_H0> <label>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    seed = int(sys.argv[2])
    count_on = int(sys.argv[3])
    TP_on = int(sys.argv[4])
    ref_on = int(sys.argv[5])
    only_H0 = int(sys.argv[6])
    label = sys.argv[7]
    
    run_single_job(dataset_name, seed, count_on, TP_on, ref_on, only_H0, label)
