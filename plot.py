import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

import sys
import os

SAVE_TO_PNG = False


# COLUMN HEADER FORMAT : index,build,test_index,PACKET_SIZE,PACKET_RATE,RX-RATE-PPS,RX-RATE-GBPS,LOSE-RATE,AVG-LAT,RX-GOODPUT-MBPS-PKTGEN,RX-GOODPUT-GBPS-PKTGEN,run_index

FONT_SIZE = 22
BORDER_WIDTH = 3

def set_plot_style():
    """Set consistent style for all plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['axes.labelsize'] = FONT_SIZE
    mpl.rcParams['axes.titlesize'] = FONT_SIZE + 2
    mpl.rcParams['xtick.labelsize'] = FONT_SIZE
    mpl.rcParams['ytick.labelsize'] = FONT_SIZE
    mpl.rcParams['legend.fontsize'] = FONT_SIZE
    mpl.rcParams['figure.titlesize'] = FONT_SIZE
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['lines.markersize'] = 12
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = '--'
    mpl.rcParams['grid.alpha'] = 0.7
    # change the font to a more readable one
    mpl.rcParams['font.family'] = 'sans-serif'

def open_file(filename):
    """
    Open a file and return the data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(filename, sep=",", header=0)
        # Replace missing values with 0
        data.fillna(0, inplace=True)
        return data
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None


def plot_data(df, x_col, y_col, group_col, x_legend, y_legend, output_file, filename, all_xticks=True):
    """
    Plot data from a DataFrame with different markers per group and standard deviation error bars.
    Parameters:
        df: DataFrame containing the data
        x_col: Column name for x-axis values
        y_col: Column name for y-axis values
        group_col: Column name for grouping
        x_legend: X-axis label
        y_legend: Y-axis label
        output_file: Base name for the output file
        filename: Base filename for directory structure
        all_xticks: boolean, if True shows a tick for each x value (default: True)
    """
    # Apply consistent styling
    set_plot_style()

    plt.figure(figsize=(12, 8))
    markers = ['o', 's', 'D', '^', '+', 'x', 'p', '*', 'h', '<', '>', 'H', 'v', 'd']
    colors = plt.cm.tab10.colors

    ax = plt.gca()
    ax.tick_params(axis='x', which='both', direction='in', length=5, width=BORDER_WIDTH, top=True, bottom=True)
    ax.tick_params(axis='y', which='both', direction='in', length=5, width=BORDER_WIDTH, left=True, right=True)

    # Get unique x values for tick positioning
    unique_x_vals = sorted(df[x_col].unique())
    unique_x = np.arange(len(unique_x_vals))

    for i, group in enumerate(sorted(df[group_col].unique())):
        subset = df[df[group_col] == group]
        # Group by x_col and calculate mean and std for y_col
        grouped = subset.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()

        # Sort by x_col to ensure proper line connections
        grouped = grouped.sort_values(by=x_col)

        # Replace any remaining NaN values with 0
        grouped.fillna(0, inplace=True)

        # Map x values to positions
        x_positions = [list(unique_x_vals).index(x) for x in grouped[x_col]]

        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]

        # Plot mean values with connecting lines and markers at each point
        plt.plot(x_positions, grouped['mean'],
                 label=group, marker=marker, color=color, linestyle='-', markersize=8)

        # Add error bars for standard deviation
        plt.errorbar(x_positions, grouped['mean'],
                     yerr=grouped['std'],
                     color=color,
                     alpha=0.7,
                     fmt='none',
                     capsize=5,
                     elinewidth=1.5)

    # plt.title(f"{y_col} vs {x_col}")
    plt.legend(title=group_col, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fancybox=True, title_fontsize=FONT_SIZE)
    plt.ylim(0, None)  # Set y-axis limit to start from 0
    plt.xlabel(x_legend, fontweight='bold')
    plt.ylabel(y_legend, fontweight='bold')

    # Always set x-ticks
    plt.xticks(unique_x, unique_x_vals)

    # Add a thin border
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(BORDER_WIDTH)

    plt.tight_layout()
    plt.savefig(f"output/{filename}/{output_file}.pdf", bbox_inches='tight', dpi=300)
    # Also save as PNG for easier viewing
    if SAVE_TO_PNG:
        plt.savefig(f"output/{filename}/{output_file}.png", bbox_inches='tight', dpi=300)
    plt.close()


def plot_bar_comparison(df, x_col, y_col, group_col, x_legend, y_legend, output_file, filename):
    """
    Create a grouped bar plot to compare different implementations.

    Parameters:
        df: DataFrame containing the data
        x_col: Column name for x-axis categories (e.g., 'PACKET_SIZE')
        y_col: Column name for y-axis values (e.g., 'RX-RATE-PPS')
        group_col: Column name for grouping (e.g., 'Implementation')
        x_legend: X-axis label
        y_legend: Y-axis label
        output_file: Base name for the output file
        filename: Base filename for directory structure
    """
    # Apply consistent styling
    set_plot_style()

    # Create figure with appropriate size
    plt.figure(figsize=(12, 8))

    # Get unique x values and group values
    x_values = sorted(df[x_col].unique())
    groups = sorted(df[group_col].unique())

    # Number of groups and bar width
    n_groups = len(groups)
    bar_width = 0.8 / n_groups

    # Colors for the bars - use same colors as line plots for consistency
    colors = plt.cm.tab10.colors[:n_groups]

    # Initialize positions
    index = np.arange(len(x_values))

    # Define hatches for print-friendly distinction
    hatches = ['/', '\\', 'x', '+', '*', 'o', 'O', '.', '|', '-']

    # Create the grouped bars
    ax = plt.gca()
    # Tweak tick appearance
    ax.tick_params(axis='x', which='both', direction='in', length=5, width=BORDER_WIDTH, top=True, bottom=False)
    ax.tick_params(axis='y', which='both', direction='in', length=5, width=BORDER_WIDTH, left=True, right=True)
    for i, group in enumerate(groups):
        # Filter data for this group
        group_data = df[df[group_col] == group]

        # Group by x_col and calculate mean and std for y_col
        stats = group_data.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()

        # Replace any NaN values with 0
        stats.fillna(0, inplace=True)

        # Reorder according to sorted x_values
        ordered_stats = pd.DataFrame(index=range(len(x_values)), columns=['mean', 'std'])
        for j, x_val in enumerate(x_values):
            if x_val in stats[x_col].values:
                row = stats[stats[x_col] == x_val]
                ordered_stats.loc[j, 'mean'] = row['mean'].values[0]
                ordered_stats.loc[j, 'std'] = row['std'].values[0]
            else:
                ordered_stats.loc[j, 'mean'] = 0
                ordered_stats.loc[j, 'std'] = 0

        # Position for this group's bars
        pos = index + i * bar_width - (n_groups - 1) * bar_width / 2

        # Create the bars with error bars and hatches
        bars = ax.bar(pos, ordered_stats['mean'], bar_width,
                      yerr=ordered_stats['std'],
                      color=colors[i],
                      label=group,
                      capsize=4,
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=1,
                      hatch=hatches[i % len(hatches)])  # Add hatch pattern

        # Add value labels on top of bars
        # for bar in bars:
        #     height = bar.get_height()
        #     if height > 0:  # Only add text if bar has height
        #         # Format value based on magnitude
        #         if height >= 1_000_000:
        #             value_text = f'{height / 1_000_000:.2f}M'
        #         elif height >= 1000:
        #             value_text = f'{height / 1000:.2f}K'
        #         else:
        #             value_text = f'{height:.2f}'
        #
        #         ax.text(bar.get_x() + bar.get_width() / 2., height + ordered_stats['std'].max() * 0.2,
        #                 value_text, ha='center', va='bottom', rotation=45,
        #                 fontsize=8, fontweight='bold')

    # Set labels and title
    plt.xlabel(x_legend, fontweight='bold')
    plt.ylabel(y_legend, fontweight='bold')
    # plt.title(f'Comparison of {y_col} by {group_col} for different {x_col} values', pad=20)

    # Set x-ticks at the center of each group
    plt.xticks(index, x_values)

    for tick in plt.gca().get_xticklabels():
        tick.set_visible(True)
        tick.set_color('black')  # Or any visible color
        tick.set_alpha(1.0)


    # Add a legend
    plt.legend(title=group_col, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=n_groups, fancybox=True, title_fontsize=FONT_SIZE)

    # Set y-axis to start at 0 and use reasonable number of ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    plt.ylim(bottom=0)

    # Add a thin border
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(BORDER_WIDTH)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"output/{filename}/{output_file}_bar.pdf", bbox_inches='tight', dpi=300)
    if SAVE_TO_PNG:
        plt.savefig(f"output/{filename}/{output_file}_bar.png", bbox_inches='tight', dpi=300)
    plt.close()


def plot_aggregate_comparison(df, metric_col, group_col, output_file, filename):
    """
    Create an aggregate comparison plot showing the average performance
    of each implementation across all packet sizes and rates.

    Parameters:
        df: DataFrame containing the data
        metric_col: Column name for the metric to aggregate (e.g., 'RX-RATE-PPS')
        group_col: Column name for grouping (e.g., 'Implementation')
        output_file: Base name for the output file
        filename: Base filename for directory structure
    """
    # Apply consistent styling
    set_plot_style()

    plt.figure(figsize=(10, 6))

    # Aggregate data by implementation
    agg_data = df.groupby(group_col)[metric_col].agg(['mean', 'std']).reset_index()

    # Replace any NaN values with 0
    agg_data.fillna(0, inplace=True)

    # Sort by mean value for better visualization
    agg_data = agg_data.sort_values('mean', ascending=False)

    # Colors matching the line plot colors
    colors = plt.cm.tab10.colors[:len(agg_data)]

    # Create horizontal bars
    ax = plt.gca()
    bars = ax.barh(agg_data[group_col], agg_data['mean'],
                   xerr=agg_data['std'],
                   color=colors,
                   capsize=5,
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=1)

    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width > 0:
            # Format value based on magnitude
            if width >= 1_000_000:
                value_text = f'{width / 1_000_000:.2f}M'
            elif width >= 1000:
                value_text = f'{width / 1000:.2f}K'
            else:
                value_text = f'{width:.2f}'

            ax.text(width + agg_data['std'].max() * 0.2, bar.get_y() + bar.get_height() / 2,
                    value_text, va='center', ha='left', fontweight='bold')

    metric_name = metric_col.replace('-', ' ')
    # plt.title(f'Aggregate {metric_name} by {group_col}')
    plt.xlabel(metric_name, fontweight='bold')
    plt.ylabel(group_col, fontweight='bold')

    # Set x-axis to start at 0
    plt.xlim(0, None)

    # Add a thin border
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(f"output/{filename}/aggregate_{output_file}.pdf", bbox_inches='tight', dpi=300)
    if SAVE_TO_PNG:
        plt.savefig(f"output/{filename}/aggregate_{output_file}.png", bbox_inches='tight', dpi=300)
    plt.close()


def plot_heatmap(df, x_col, y_col, value_col, group_col, group_val, x_legend, y_legend, output_file, filename):
    """
    Create a heatmap showing the performance across different packet sizes and rates.

    Parameters:
        df: DataFrame containing the data
        x_col: Column for x-axis (e.g., 'PACKET_SIZE')
        y_col: Column for y-axis (e.g., 'PACKET_RATE')
        value_col: Column for cell values (e.g., 'RX-RATE-PPS')
        group_col: Column for filtering (e.g., 'Implementation')
        group_val: Value to filter by (e.g., 'VECTOR_AVX')
        x_legend: X-axis label
        y_legend: Y-axis label
        output_file: Base name for the output file
        filename: Base filename for directory structure
    """
    # Apply consistent styling
    set_plot_style()

    # Filter data for the specific implementation
    filtered_df = df[df[group_col] == group_val]

    # Create pivot table for the heatmap
    pivot_df = filtered_df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc='mean')

    # Replace NaN values with 0
    pivot_df.fillna(0, inplace=True)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Create heatmap
    im = plt.imshow(pivot_df, cmap='viridis', aspect='auto', interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(value_col, rotation=270, labelpad=20, fontweight='bold')

    # Set axis ticks and labels
    plt.xticks(np.arange(len(pivot_df.columns)), pivot_df.columns, rotation=45)
    plt.yticks(np.arange(len(pivot_df.index)), pivot_df.index)

    plt.xlabel(x_legend, fontweight='bold')
    plt.ylabel(y_legend, fontweight='bold')
    # plt.title(f'{value_col} for {group_val} across {x_col} and {y_col}')

    # Add text annotations in the cells
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            # Format based on magnitude
            if value == 0:
                text = '0'
            elif value >= 1_000_000:
                text = f'{value / 1_000_000:.1f}M'
            elif value >= 1000:
                text = f'{value / 1000:.1f}K'
            else:
                text = f'{value:.1f}'

            plt.text(j, i, text, ha='center', va='center',
                     color='black' if value > pivot_df.mean().mean() else 'white',
                     fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"output/{filename}/heatmap_{group_val}_{output_file}.pdf", bbox_inches='tight', dpi=300)
    if SAVE_TO_PNG:
        plt.savefig(f"output/{filename}/heatmap_{group_val}_{output_file}.png", bbox_inches='tight', dpi=300)
    plt.close()


def plot_aggregate_grid(df, metric_cols, group_col, output_file, filename):
    """
    Create a grid of aggregate plots for multiple metrics.

    Parameters:
        df: DataFrame containing the data
        metric_cols: List of column names for metrics to aggregate
        group_col: Column name for grouping (e.g., 'Implementation')
        output_file: Base name for the output file
        filename: Base filename for directory structure
    """
    # Apply consistent styling
    set_plot_style()

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    n_metrics = len(metric_cols)

    # Determine grid layout
    if n_metrics <= 3:
        n_rows, n_cols = 1, n_metrics
    else:
        n_rows = (n_metrics + 1) // 2
        n_cols = 2

    # Colors matching the line plot colors
    groups = df[group_col].unique()
    colors = plt.cm.tab10.colors[:len(groups)]

    for i, metric in enumerate(metric_cols):
        ax = plt.subplot(n_rows, n_cols, i + 1)

        # Aggregate data by implementation
        agg_data = df.groupby(group_col)[metric].agg(['mean', 'std']).reset_index()

        # Replace any NaN values with 0
        agg_data.fillna(0, inplace=True)

        # Sort by mean value for better visualization
        agg_data = agg_data.sort_values('mean', ascending=False)

        # Create horizontal bars
        bars = ax.barh(agg_data[group_col], agg_data['mean'],
                       xerr=agg_data['std'],
                       color=colors[:len(agg_data)],
                       capsize=5,
                       alpha=0.8,
                       edgecolor='black',
                       linewidth=1)

        # Add value labels on the bars
        for j, bar in enumerate(bars):
            width = bar.get_width()
            if width > 0:
                # Format value based on magnitude
                if width >= 1_000_000:
                    value_text = f'{width / 1_000_000:.2f}M'
                elif width >= 1000:
                    value_text = f'{width / 1000:.2f}K'
                else:
                    value_text = f'{width:.2f}'

                ax.text(width + agg_data['std'].max() * 0.2, bar.get_y() + bar.get_height() / 2,
                        value_text, va='center', ha='left', fontweight='bold', fontsize=8)

        metric_name = metric.replace('-', ' ')
        ax.set_title(metric_name)
        ax.set_xlabel(metric_name)
        if i % n_cols == 0:  # Only add y-label on leftmost plots
            ax.set_ylabel(group_col, fontweight='bold')

        # Set x-axis to start at 0
        ax.set_xlim(0, None)

    plt.suptitle(f'Aggregate Performance Metrics by {group_col}', fontsize=FONT_SIZE, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    plt.savefig(f"output/{filename}/aggregate_grid_{output_file}.pdf", bbox_inches='tight', dpi=300)
    if SAVE_TO_PNG:
        plt.savefig(f"output/{filename}/aggregate_grid_{output_file}.png", bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    # If there is an argument, it is the base name of the files to plot, otherwise, use the latest filename
    if len(sys.argv) > 1:
        base_filename = sys.argv[1]
    else:
        # Get the latest set of files with the required suffixes
        files = [f for f in os.listdir('./results/') if f.endswith('.csv')]
        if not files:
            print("No files found.")
            sys.exit(1)

        # Extract base filenames (without suffixes)
        base_filenames = set()
        for f in files:
            for suffix in ['LL_NOAVX.csv', 'VECTOR_AVX.csv', 'VECTOR_NOAVX.csv']:
                if f.endswith(suffix):
                    base_filenames.add(f[:-len(suffix)])

        if not base_filenames:
            print("No files with required suffixes found.")
            sys.exit(1)

        # Get the latest base filename
        base_filenames = sorted(list(base_filenames))
        base_filename = base_filenames[-1]

    print(f"Using base filename: {base_filename}")

    # Define the required suffixes
    suffixes = ['LL_NOAVX.csv', 'VECTOR_AVX.csv', 'VECTOR_NOAVX.csv']

    # Create a combined dataframe with an additional column for the implementation type
    combined_df = None

    for suffix in suffixes:
        full_filename = f"{base_filename}{suffix}"
        file_path = f"results/{full_filename}"

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping.")
            continue

        # Open the file
        df = open_file(file_path)

        # Check if the file was opened successfully
        if df is None:
            print(f"Error opening file {file_path}, skipping.")
            continue

        # Add implementation type column based on suffix
        implementation = suffix.replace('.csv', '')
        df['Implementation'] = implementation

        # Replace any remaining NaN values with 0
        df.fillna(0, inplace=True)

        # Append to combined dataframe
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    if combined_df is None or combined_df.empty:
        print("No valid data found in any of the files.")
        sys.exit(1)

    # Ensure all missing values are replaced with 0
    combined_df.fillna(0, inplace=True)

    combined_df["PACKET_RATE_MBPS"] = combined_df["PACKET_RATE"] * 25000 / 100
    combined_df["PACKET_RATE_GBPS"] = combined_df["PACKET_RATE_MBPS"] / 1000
    combined_df["RX-RATE-GBPS"] = combined_df["RX-RATE-MBPS"] / 1000
    # filter out lines where LOSE-RATE is > 0.5
    print("NOT Filtering out lines where LOSE-RATE is > 0.1")
    print("PREVIOUS N° OF LINES: ", len(combined_df))
    # combined_df = combined_df[combined_df["LOSE-RATE"] <= 0.5]
    # combined_df = combined_df[combined_df["AVG-LAT"] <= 1000]
    # combined_df = combined_df[combined_df["AVG-LAT"] >= 1]
    print("AFTER N° OF LINES: ", len(combined_df))

    # Create output directory
    output_dir = f"output/{base_filename}"
    os.makedirs("output", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Define metrics for plotting
    metrics = ['RX-RATE-PPS', 'RX-RATE-GBPS', 'LOSE-RATE', 'AVG-LAT']

    # Create aggregate plots for each metric
    for metric in metrics:
        plot_aggregate_comparison(combined_df, metric, 'Implementation', metric.lower(), base_filename)

    # Create an aggregate grid showing all metrics
    plot_aggregate_grid(combined_df, metrics, 'Implementation', 'all_metrics', base_filename)

    # Create heatmaps for each implementation and metric
    for impl in combined_df['Implementation'].unique():
        for metric in metrics:
            plot_heatmap(combined_df, 'PACKET_SIZE', 'PACKET_RATE_GBPS', metric,
                         'Implementation', impl, 'Packet Size (bytes)', 'Packet Rate (gbps)',
                         metric.lower(), base_filename)

    # Create plot for each packet rate, comparing implementations
    packet_rates = combined_df['PACKET_RATE_GBPS'].unique()

    for rate in packet_rates:
        rate_df = combined_df[combined_df['PACKET_RATE_GBPS'] == rate]

        # Plot RX-RATE-PPS vs PACKET_SIZE for all implementations at this rate
        plot_data(rate_df, "PACKET_SIZE", "RX-RATE-PPS", "Implementation",
                  "Packet Size (bytes)", "RX Rate (PPS)",
                  f"rx_rate_pps_rate_{rate}", base_filename)

        # Add bar plot version
        plot_bar_comparison(rate_df, "PACKET_SIZE", "RX-RATE-PPS", "Implementation",
                            "Packet Size (bytes)", "RX Rate (PPS)",
                            f"rx_rate_pps_rate_{rate}", base_filename)

        # Plot RX-RATE-GBPS vs PACKET_SIZE
        plot_data(rate_df, "PACKET_SIZE", "RX-RATE-GBPS", "Implementation",
                  "Packet Size (bytes)", "RX Rate (Gbps)",
                  f"rx_rate_gbps_rate_{rate}", base_filename)

        # Add bar plot version
        plot_bar_comparison(rate_df, "PACKET_SIZE", "RX-RATE-GBPS", "Implementation",
                            "Packet Size (bytes)", "RX Rate (Gbps)",
                            f"rx_rate_gbps_rate_{rate}", base_filename)

        # Plot LOSE-RATE vs PACKET_SIZE
        plot_data(rate_df, "PACKET_SIZE", "LOSE-RATE", "Implementation",
                  "Packet Size (bytes)", "Loss Rate",
                  f"loss_rate_rate_{rate}", base_filename)

        # Add bar plot version
        plot_bar_comparison(rate_df, "PACKET_SIZE", "LOSE-RATE", "Implementation",
                            "Packet Size (bytes)", "Loss Rate",
                            f"loss_rate_rate_{rate}", base_filename)

        # Plot AVG-LAT vs PACKET_SIZE
        plot_data(rate_df, "PACKET_SIZE", "AVG-LAT", "Implementation",
                  "Packet Size (bytes)", "Average Latency (us)",
                  f"avg_lat_rate_{rate}", base_filename)

        # Add bar plot version
        plot_bar_comparison(rate_df, "PACKET_SIZE", "AVG-LAT", "Implementation",
                            "Packet Size (bytes)", "Average Latency (us)",
                            f"avg_lat_rate_{rate}", base_filename)

    # Create a direct comparison plot for each packet size, showing performance across implementations
    packet_sizes = combined_df['PACKET_SIZE'].unique()
    for size in packet_sizes:
        size_df = combined_df[combined_df['PACKET_SIZE'] == size]

        # Create plots comparing implementations at this packet size
        plot_data(size_df, "PACKET_RATE_GBPS", "RX-RATE-PPS", "Implementation",
                  "Packet Rate (Gbps)", "RX Rate (PPS)",
                  f"impl_comparison_pps_size_{size}", base_filename)

        plot_data(size_df, "PACKET_RATE_GBPS", "RX-RATE-GBPS", "Implementation",
                  "Packet Rate (Gbps)", "RX Rate (Gbps)",
                  f"impl_comparison_gbps_size_{size}", base_filename)

        plot_data(size_df, "PACKET_RATE_GBPS", "LOSE-RATE", "Implementation",
                  "Packet Rate (Gbps)", "Loss Rate",
                  f"impl_comparison_loss_size_{size}", base_filename)

        plot_data(size_df, "PACKET_RATE_GBPS", "AVG-LAT", "Implementation",
                  "Packet Rate (Gbps)", "Average Latency (us)",
                  f"impl_comparison_avg_lat_size_{size}", base_filename)

        # Bar plot versions
        plot_bar_comparison(size_df, "PACKET_RATE_GBPS", "RX-RATE-PPS", "Implementation",
                            "Packet Rate (Gbps)", "RX Rate (PPS)",
                            f"impl_comparison_pps_size_{size}", base_filename)

        plot_bar_comparison(size_df, "PACKET_RATE_GBPS", "RX-RATE-GBPS", "Implementation",
                            "Packet Rate (Gbps)", "RX Rate (Gbps)",
                            f"impl_comparison_gbps_size_{size}", base_filename)

        plot_bar_comparison(size_df, "PACKET_RATE_GBPS", "LOSE-RATE", "Implementation",
                            "Packet Rate (Gbps)", "Loss Rate",
                            f"impl_comparison_loss_size_{size}", base_filename)

        plot_bar_comparison(size_df, "PACKET_RATE_GBPS", "AVG-LAT", "Implementation",
                            "Packet Rate (Gbps)", "Average Latency (us)",
                            f"impl_comparison_avg_lat_size_{size}", base_filename)

    plot_bar_comparison(combined_df, "PACKET_SIZE", "RX-RATE-GBPS", "Implementation",
                        "Packet Size (bytes)", "RX Rate (Gbps)",
                        f"rx_rate_gbps_all", base_filename)

    plot_bar_comparison(combined_df, "PACKET_SIZE", "RX-RATE-PPS", "Implementation",
                        "Packet Size (bytes)", "RX Rate (PPS)",
                        f"rx_rate_pps_all", base_filename)

    plot_bar_comparison(combined_df, "PACKET_SIZE", "LOSE-RATE", "Implementation",
                        "Packet Size (bytes)", "Loss Rate",
                        f"loss_rate_all", base_filename)

    plot_bar_comparison(combined_df, "PACKET_RATE_GBPS", "RX-RATE-GBPS", "Implementation",
                        "Packet Rate (Gbps)", "RX Rate (Gbps)",
                        f"rx_rate_gbps_all_rate", base_filename)

    # Also create plots comparing all packet rates for each implementation
    for impl in combined_df['Implementation'].unique():
        impl_df = combined_df[combined_df['Implementation'] == impl]

        # Plot RX-RATE-PPS vs PACKET_SIZE
        plot_data(impl_df, "PACKET_SIZE", "RX-RATE-PPS", "PACKET_RATE_GBPS",
                  "Packet Size (bytes)", "RX Rate (PPS)",
                  f"rx_rate_pps_{impl}", base_filename)

        # Plot RX-RATE-GBPS vs PACKET_SIZE
        plot_data(impl_df, "PACKET_SIZE", "RX-RATE-GBPS", "PACKET_RATE_GBPS",
                  "Packet Size (bytes)", "RX Rate (Gbps)",
                  f"rx_rate_gbps_{impl}", base_filename)

        # Plot LOSE-RATE vs PACKET_SIZE
        plot_data(impl_df, "PACKET_SIZE", "LOSE-RATE", "PACKET_RATE_GBPS",
                  "Packet Size (bytes)", "Loss Rate",
                  f"loss_rate_{impl}", base_filename)
