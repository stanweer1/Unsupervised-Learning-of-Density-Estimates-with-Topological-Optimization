import pandas as pd
import numpy as np
import glob
import os


def analyze_results(results_dir='results'):
    """
    Loads all simulation CSVs from a directory, combines them,
    and prints the mean and std dev for each metric.

    Special rule for MNIST:
        - Split MNIST results by digit = seed % 10.
    """

    # -------------------------------------------------------------
    # Load CSVs
    # -------------------------------------------------------------
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files in {results_dir}")
        return

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # -------------------------------------------------------------
    # Add digit column for MNIST ONLY
    # -------------------------------------------------------------
    df["digit"] = np.where(df["dataset"] == "MNIST",
                           df["seed"] % 10,
                           np.nan)

    # Separate MNIST / non-MNIST
    df_mnist = df[df["dataset"] == "MNIST"].copy()
    df_other = df[df["dataset"] != "MNIST"].copy()

    # -------------------------------------------------------------
    # Grouping
    # -------------------------------------------------------------
    results_tables = {}

    # MNIST grouping: (dataset="MNIST", digit, method)
    if not df_mnist.empty:
        grouped_mnist = df_mnist.groupby(["digit", "method"])
        mean_mnist = grouped_mnist[["time"]].mean()
        std_mnist  = grouped_mnist[["time"]].std()

        results_tables["mnist"] = mean_mnist.merge(
            std_mnist, left_index=True, right_index=True,
            suffixes=("_mean", "_std")
        )

    # -------------------------------------------------------------
    # Pretty formatting
    # -------------------------------------------------------------
    for key in results_tables:
        rt = results_tables[key]
        rt["time"] = rt.apply(lambda r: f"{r['time_mean']:.8f} ± {r['time_std']:.8f}", axis=1)
        results_tables[key] = rt

    method_order = [
        'Scott', 'Silverman', 'NRR', 'ML-CV',
        'LSCV', 'BCV', 'BotevProj',
        'PluginDiag', 'TDA', 'ISJ'
    ]

    print("\n--- Simulation Results (Mean ± Std Dev) ---\n")

    # -------------------------------------------------------------
    # Print MNIST datasets
    # -------------------------------------------------------------
    if "mnist" in results_tables:
        print("\nDataset: MNIST (digit = seed % 10)")

        rt = results_tables["mnist"]
        digits = sorted(rt.index.get_level_values("digit").unique())

        for digit in digits:
            print(f"\n  Digit: {int(digit)}")

            try:
                subset = rt.xs(digit, level="digit")
            except KeyError:
                continue

            subset = subset[["time"]].reindex(method_order, fill_value="N/A")
            print(subset.to_string())


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    analyze_results()