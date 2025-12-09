import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ===============================================
# 1. Load all NPZ results from hyperparameter_sensitivity/
# ===============================================

def load_sensitivity_results(results_dir="hyperparameter_sensitivity"):
    files = glob.glob(os.path.join(results_dir, "*.npy"))
    if not files:
        print("No results found.")
        return None

    rows = []

    for f in files:
        fname = os.path.basename(f).replace(".npy", "")
        parts = fname.split("_")

        if len(parts) < 6:
            print("Skipping malformed filename:", fname)
            continue

        # Parse from the END (robust to underscores in dataset name)
        try:
            grid = int(parts[-1])
            reg  = float(parts[-2])
            tp   = float(parts[-3])
            pe   = float(parts[-4])
            seed = int(parts[-5])
            dataset = "_".join(parts[:-5])
        except Exception as e:
            print(f"Error parsing {fname}: {e}")
            continue

        arr = np.load(f)

        # Initialize defaults
        kld = emd = bw = np.nan

        # Parse the array structure
        if arr.size == 3:
            kld, emd, bw = arr
        elif arr.size == 2:
            kld, emd = arr   # bw stays nan
        else:
            print(f"Invalid array size in {fname}: shape={arr.shape}")
            continue   # skip only invalid-sized arrays

        if (not np.isfinite(kld)):
            print(f"Skipping due to invalid KLD: {fname}")
            continue


        rows.append({
            "dataset": dataset,
            "seed": seed,
            "pe": pe,
            "tp": tp,
            "reg": reg,
            "grid": grid,
            "kld": kld,
            "emd": emd,
            "bw": bw
        })

    df = pd.DataFrame(rows)
    print("Loaded", len(df), "rows.")

    return df

# ===============================================
# 2. Plotting helpers
# ===============================================

def plot_sensitivity(df, param, fixed_params, param_label, outfile_prefix):
    """
    Creates two plots:
        1. KLD vs param
        2. EMD vs param
    saved as: <outfile_prefix>_kld.png and <outfile_prefix>_emd.png
    """

    metrics_to_plot = {
        "kld": ("KLD", f"{outfile_prefix}_kld.png"),
        "emd": ("EMD", f"{outfile_prefix}_emd.png"),
    }

    datasets = df.dataset.unique()
    print(datasets)

    for metric, (ylabel, outfile) in metrics_to_plot.items():

        plt.figure(figsize=(6, 4))

        for ds in datasets:
            df_ds = df[df.dataset == ds]

            # Apply fixed parameter filters
            for key, val in fixed_params.items():
                df_ds = df_ds[df_ds[key] == val]

            # Group by sweeping parameter
            g = df_ds.groupby(param)[metric]

            means = g.mean()
            stds = g.std()
            xs = means.index.values

            # Skip empty lines (in case some param combos do not exist)
            if len(xs) == 0:
                continue

            plt.errorbar(
                xs, 
                means.values, 
                yerr=stds.values, 
                marker='o', 
                capsize=4, 
                label=ds[:-5]
            )

        plt.xlabel(param_label)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs {param_label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()

        print("Saved plot:", outfile)


# ===============================================
# 3. Main analysis logic
# ===============================================

def analyze_sensitivity(results_dir="hyperparameter_sensitivity"):

    df = load_sensitivity_results(results_dir)
    if df is None:
        return

    # Baseline values used in your SLURM script
    BASE_PE = 1
    BASE_TP = 1
    BASE_REG = 0
    BASE_GRID = 200

    os.makedirs("sensitivity_plots", exist_ok=True)

    # Sweep 1: α_PE
    plot_sensitivity(
        df,
        param="pe",
        fixed_params={"tp": BASE_TP, "reg": BASE_REG, "grid": BASE_GRID},
        param_label=r"$\alpha_{count}$",
        outfile_prefix="sensitivity_plots/sensitivity_pe"
    )
    
    # Sweep 2: α_TP
    plot_sensitivity(
        df,
        param="tp",
        fixed_params={"pe": BASE_PE, "reg": BASE_REG, "grid": BASE_GRID},
        param_label=r"$\alpha_{TP}$",
        outfile_prefix="sensitivity_plots/sensitivity_tp"
    )
    
    # Sweep 3: λ
#    plot_sensitivity(
#        df,
#        param="reg",
#        fixed_params={"pe": BASE_PE, "tp": BASE_TP, "grid": BASE_GRID},
#        param_label=r"$\lambda$",
#        outfile_prefix="sensitivity_plots/sensitivity_reg"
#    )
    
    # Sweep 4: grid
    plot_sensitivity(
        df,
        param="grid",
        fixed_params={"pe": BASE_PE, "tp": BASE_TP, "reg": BASE_REG},
        param_label="Grid Resolution",
        outfile_prefix="sensitivity_plots/sensitivity_grid"
    )

    print("\nAll sensitivity plots saved in: sensitivity_plots/")


# ===============================================
# ENTRY POINT
# ===============================================

if __name__ == "__main__":
    analyze_sensitivity()
