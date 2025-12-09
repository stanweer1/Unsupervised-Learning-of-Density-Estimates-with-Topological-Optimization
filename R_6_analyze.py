import pandas as pd
import numpy as np
import glob
import os

def analyze_results(results_dir='results'):
    """
    Loads all individual simulation CSVs from a directory, combines them,
    and prints the mean and std dev for each metric.
    """
    
    # 1. Find all CSV files in the results directory
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in '{results_dir}'.")
        print("Please run '5_run_job.py' first to generate results.")
        return

    print(f"Found {len(csv_files)} result files. Loading and concatenating...")

    # 2. Read and concatenate all files into a single DataFrame
    df_list = []
    for f in csv_files:
        df_list.append(pd.read_csv(f))
    
    df = pd.concat(df_list, ignore_index=True)
    print("All results loaded.")

    # 3. Group by dataset and method
    grouped = df.groupby(['dataset', 'method'])
    
    # 4. Calculate mean and std dev
    mean_scores = grouped[['kld', 'emd']].mean()
    std_scores = grouped[['kld', 'emd']].std()
    
    # 5. Combine into a single, printable dataframe
    results_table = mean_scores.merge(std_scores, 
                                      left_index=True, 
                                      right_index=True, 
                                      suffixes=('_mean', '_std'))
    
    # Reformat for better readability
    results_table['KLD'] = results_table.apply(lambda row: f"{row['kld_mean']:.4f} ± {row['kld_std']:.4f}", axis=1)
    results_table['EMD'] = results_table.apply(lambda row: f"{row['emd_mean']:.4f} ± {row['emd_std']:.4f}", axis=1)
    
    # Define a sort order for methods
#    method_order = ['TDA', 'ISJ', 'GridSearchCV', 'Scott', 'Silverman']
    method_order = [
        'Scott', 'Silverman', 'NRR', 'ML-CV',
        'LSCV', 'BCV', 'BotevProj',
        'PluginDiag', 'TDA', 'ISJ'
    ]
    
    print("\n--- Simulation Results (Mean ± Std Dev) ---")
    
    # 6. Print the formatted results for each dataset
    for dataset in sorted(df['dataset'].unique()):
        print(f"\nDataset: {dataset} (from {len(df[df['dataset']==dataset]['seed'].unique())} runs)")
        
        # Get results for this dataset and re-order
        table_subset = results_table.loc[dataset][['KLD', 'EMD']]
        table_subset = table_subset.reindex(method_order, fill_value='N/A')
        
        print(table_subset.to_string())

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

    # Non-MNIST grouping: (dataset, method)
    if not df_other.empty:
        grouped_other = df_other.groupby(["dataset", "method"])
        mean_other = grouped_other[["kld", "emd"]].mean()
        std_other  = grouped_other[["kld", "emd"]].std()

        results_tables["other"] = mean_other.merge(
            std_other, left_index=True, right_index=True,
            suffixes=("_mean", "_std")
        )

    # MNIST grouping: (dataset="MNIST", digit, method)
    if not df_mnist.empty:
        grouped_mnist = df_mnist.groupby(["digit", "method"])
        mean_mnist = grouped_mnist[["kld", "emd"]].mean()
        std_mnist  = grouped_mnist[["kld", "emd"]].std()

        results_tables["mnist"] = mean_mnist.merge(
            std_mnist, left_index=True, right_index=True,
            suffixes=("_mean", "_std")
        )

    # -------------------------------------------------------------
    # Pretty formatting
    # -------------------------------------------------------------
    for key in results_tables:
        rt = results_tables[key]
        rt["KLD"] = rt.apply(lambda r: f"{r['kld_mean']:.4f} ± {r['kld_std']:.4f}", axis=1)
        rt["EMD"] = rt.apply(lambda r: f"{r['emd_mean']:.4f} ± {r['emd_std']:.4f}", axis=1)
        results_tables[key] = rt

    method_order = [
        'Scott', 'Silverman', 'NRR', 'ML-CV',
        'LSCV', 'BCV', 'BotevProj',
        'PluginDiag', 'TDA', 'ISJ'
    ]

    print("\n--- Simulation Results (Mean ± Std Dev) ---\n")

    # -------------------------------------------------------------
    # Print Non-MNIST datasets
    # -------------------------------------------------------------
    if "other" in results_tables:
        rt = results_tables["other"]

        for dataset in sorted(df_other["dataset"].unique()):
            print(f"\nDataset: {dataset}")

            try:
                subset = rt.xs(dataset, level="dataset")
            except KeyError:
                print("  No results.")
                continue

            subset = subset[["KLD", "EMD"]].reindex(method_order, fill_value="N/A")
            print(subset.to_string())

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

            subset = subset[["KLD", "EMD"]].reindex(method_order, fill_value="N/A")
            print(subset.to_string())


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    analyze_results()