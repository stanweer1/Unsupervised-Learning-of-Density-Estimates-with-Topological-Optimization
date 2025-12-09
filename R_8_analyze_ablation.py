import pandas as pd
import numpy as np
import glob
import os

ABLATION_MAP = {
    "full":       (1, 1, 1, 1),
    "no-count":   (0, 1, 1, 1),
    "no-TP":      (1, 0, 1, 1),
    "no-reg":     (1, 1, 0, 1),
    "no-H0":      (1, 1, 1, 0),
    "count-only": (1, 0, 0, 1),
    "TP-only":    (0, 1, 0, 1),
    "reg-only":   (0, 0, 1, 1),
}

def analyze_ablation(results_dir="ablation_study"):
    npy_files = glob.glob(os.path.join(results_dir, "*.npy"))
    if not npy_files:
        print("No result files found.")
        return

    rows = []
    dropped = []

    for f in npy_files:
        fname = os.path.basename(f).replace(".npy", "")

        # Correct dataset extraction:
        try:
            dataset_part, rest = fname.split("_seed_")
        except ValueError:
            print(f"Malformed filename: {fname}")
            continue

        dataset = dataset_part                     # full dataset name
        seed_and_label = rest.split("_")

        seed = int(seed_and_label[0])
        label = "_".join(seed_and_label[1:])

        if label not in ABLATION_MAP:
            print(f"Unknown label in file: {fname}")
            continue

        kld, emd = np.load(f)
        if not np.isfinite(kld) or not np.isfinite(emd):
            dropped.append(fname)
            continue

        count_on, TP_on, ref_on, only_H0 = ABLATION_MAP[label]

        rows.append({
            "dataset": dataset,
            "seed": seed,
            "label": label,
            "count_on": count_on,
            "TP_on": TP_on,
            "ref_on": ref_on,
            "only_H0": only_H0,
            "kld": float(kld),
            "emd": float(emd),
        })

    if dropped:
        print("\nDropped invalid files:")
        for d in dropped:
            print(" -", d)

    df = pd.DataFrame(rows)
    print("\nLoaded rows:", len(df))

    # Grouping
    grouped = df.groupby(["dataset", "label"])
    stats = grouped[["kld", "emd"]].agg(["mean", "std"])
    stats.columns = ["_".join(col) for col in stats.columns]

    stats["KLD"] = stats.apply(lambda r: f"{r['kld_mean']:.4f} ± {r['kld_std']:.4f}", axis=1)
    stats["EMD"] = stats.apply(lambda r: f"{r['emd_mean']:.4f} ± {r['emd_std']:.4f}", axis=1)

    for dataset in sorted(df.dataset.unique()):
        print("\nDataset:", dataset)
        print(stats.loc[dataset][["KLD", "EMD"]])
        print("-" * 60)

# ===============================================
# ENTRY POINT
# ===============================================

if __name__ == "__main__":
    analyze_ablation()