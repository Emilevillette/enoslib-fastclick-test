import pandas as pd
import numpy as np

versions = ["LL_NOAVX", "VECTOR_NOAVX", "VECTOR_AVX"]


# open the files and form a unified df with the version as a column
def read_files():
    dfs = []
    for version in versions:
        with open(f"perf_data/stat/{version}.csv", "r") as f:
            df = pd.read_csv(f)
            df["version"] = version
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# read the files
df = read_files()

# convert the columns to numeric
df["Mean"] = pd.to_numeric(df["Mean"], errors="coerce")
df["StdDev"] = pd.to_numeric(df["StdDev"], errors="coerce")

print(df.head())

# example df :
"""
                 Event          Mean        StdDev   version
0               cycles  9.031421e+10  3.434489e+07  LL_NOAVX
1         instructions  1.768276e+11  3.060759e+08  LL_NOAVX
2     cache-references  9.852372e+07  1.747702e+05  LL_NOAVX
3         cache-misses  3.337046e+06  4.329042e+04  LL_NOAVX
4  branch-instructions  2.361536e+10  4.017034e+07  LL_NOAVX
"""


# for each version, compute the baseline based on LL_NOAVX, and compute the speedup of the other versions
def compute_speedup(df):
    # Create dictionaries to store baseline values for each Event (Mean and StdDev)
    baseline_mean_dict = df[df["version"] == "LL_NOAVX"].set_index("Event")["Mean"].to_dict()
    baseline_stddev_dict = df[df["version"] == "LL_NOAVX"].set_index("Event")["StdDev"].to_dict()

    # Initialize an empty list to store results
    results = []

    # Process each row and calculate speedup properly
    for _, row in df.iterrows():
        event = row["Event"]
        version = row["version"]
        mean_value = row["Mean"]
        stddev_value = row["StdDev"]

        # Get the baseline values for this specific event
        baseline_mean = baseline_mean_dict.get(event, np.nan)
        baseline_stddev = baseline_stddev_dict.get(event, np.nan)

        # Calculate speedup
        speedup = mean_value / baseline_mean if baseline_mean != 0 else np.nan
        
        # Calculate relative error for the speedup
        # For error propagation in division (a/b), the relative error is approximately
        # sqrt((δa/a)^2 + (δb/b)^2) where δa and δb are the standard deviations
        rel_error_mean = stddev_value / mean_value if mean_value != 0 else 0
        rel_error_baseline = baseline_stddev / baseline_mean if baseline_mean != 0 else 0
        rel_error_speedup = np.sqrt(rel_error_mean**2 + rel_error_baseline**2)
        
        # Convert relative error to absolute error for the speedup
        speedup_stddev = speedup * rel_error_speedup if not np.isnan(speedup) else np.nan

        # Append to results
        results.append({
            "Event": event,
            "version": version,
            "Mean": mean_value,
            "StdDev": stddev_value,
            "Speedup": speedup,
            "Speedup_StdDev": speedup_stddev
        })

    # Convert results to DataFrame
    return pd.DataFrame(results)


# compute the speedup
speedup = compute_speedup(df)

# Limit floats to 6 decimal places
pd.set_option('display.float_format', lambda x: '%.5f' % x)
print(speedup)
# save speedup to csv with 6 decimal places
speedup.to_csv("perf_data/stat/speedup.csv", index=False, float_format='%.6f')