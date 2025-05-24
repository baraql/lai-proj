import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).parent.absolute()


def parse_log_file(file_path):
    with open(PROJECT_DIR / file_path, 'r') as f:
        content = f.read()

    scale_match = re.search(r'Loading a model with scale=(\d+)', content)
    gpu_match = re.search(r'AVAILABLE GPUS: (\d+)', content)

    if not scale_match:
        print(f"Warning: Could not extract scale configuration from {file_path}, skipping this file")
        return None

    scale = int(scale_match.group(1))

    if 'no_fsdp' in file_path:
        fsdp = False
        gpus = 1  # no_fsdp always uses 1 GPU
    elif 'fsdp' in file_path:
        fsdp = True
        if not gpu_match:
            print(f"Warning: Could not extract GPUs configuration from FSDP file {file_path}, skipping this file")
            return None
        gpus = int(gpu_match.group(1))
    else:
        print(f"Warning: Could not determine FSDP configuration from {file_path}, skipping this file")
        return None

    # extract training metrics (excluding step 1 as it's often an outlier)
    metric_pattern = r'Step: (\d+) \| Loss: ([\d.]+) \| Tokens per second: ([\d.]+) \| Training tokens per second \(%\): ([\d.]+) \| MFU \(%\): ([\d.]+) \| TFLOPs: ([\d.]+)'
    matches = re.findall(metric_pattern, content)

    if not matches:
        print(f"Warning: No training metrics found in {file_path}")
        return None

    # convert to DataFrame and filter out first step
    df = pd.DataFrame(matches, columns=['step', 'loss', 'tokens_per_sec', 'training_tokens_pct', 'mfu_pct', 'tflops'])
    df = df.astype(float)
    df = df[df['step'] > 1]  # exclude first step which is often an outlier

    if df.empty:
        print(f"Warning: No valid training steps found in {file_path}")
        return None

    result = {
        'scale': scale,
        'gpus': gpus,
        'fsdp': fsdp,
        'avg_tokens_per_sec': df['tokens_per_sec'].mean(),
        'avg_training_tokens_pct': df['training_tokens_pct'].mean(),
        'avg_mfu_pct': df['mfu_pct'].mean(),
        'avg_tflops': df['tflops'].mean(),
        'num_step_entries': len(df),
        'file_path': str(file_path)
    }

    return result


def parse_all_logs(log_file_paths):
    results = []

    for file_path in log_file_paths:
        print(f"Processing {file_path}...")
        result = parse_log_file(file_path)
        if result:
            results.append(result)

    if not results:
        print("No valid results found!")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    print(f"\nSuccessfully parsed {len(df)} log files")
    print(f"Configurations found:")
    print(df[['scale', 'gpus', 'fsdp', 'num_step_entries']].to_string(index=False))

    return df


def plot_single_metric(df, metric, title, ax=None, save_plots=False, output_dir=None):
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    # create unique configurations combining GPU count and FSDP
    df['config'] = df['gpus'].astype(str) + ' GPU' + (df['gpus'] > 1).map({True: 's', False: ''}) + ' (' + df[
        'fsdp'].map({True: 'FSDP', False: 'no FSDP'}) + ')'
    unique_configs = sorted(df['config'].unique())

    # generate colors for each configuration
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_configs)))
    color_map = dict(zip(unique_configs, colors))

    # plot each configuration
    for config in unique_configs:
        config_data = df[df['config'] == config].sort_values('scale')  # sort by scale for proper line connection

        ax.scatter(config_data['scale'], config_data[metric],
                   c=[color_map[config]],
                   label=config,
                   s=100, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)

        if len(config_data) > 1:
            ax.plot(config_data['scale'], config_data[metric],
                    color=color_map[config], alpha=0.6, linewidth=2,
                    linestyle='-', marker='o', markersize=0, zorder=2)

    ax.set_xlabel('Model Scale')
    ax.set_ylabel(title)
    ax.set_title(f'{title} vs Model Scale')
    ax.legend()
    ax.grid(True, alpha=0.3, zorder=1)

    if all(df['scale'] == df['scale'].astype(int)):
        ax.set_xticks(sorted(df['scale'].unique()))

    if save_plots and output_dir and ax == plt.gca():  # Only save if this is a standalone plot
        safe_filename = metric.replace('_', '_').replace('%', 'pct')
        plt.savefig(f'{output_dir}/{safe_filename}.png', dpi=300, bbox_inches='tight')
        print(f"Individual plot saved to {output_dir}/{safe_filename}.png")

    return ax


def create_visualizations(df, save_plots=True, output_dir=PROJECT_DIR / 'plots'):
    if df.empty:
        print("No data to visualize!")
        return

    if save_plots:
        Path(output_dir).mkdir(exist_ok=True)

    metrics = {
        'avg_tokens_per_sec': 'Average Tokens per Second',
        'avg_training_tokens_pct': 'Average Training Tokens per Second (%)',
        'avg_mfu_pct': 'Average MFU (%)',
        'avg_tflops': 'Average TFLOPs'
    }

    plt.style.use('default')
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(metrics.items()):
        plot_single_metric(df, metric, title, ax=axes[idx])

    plt.tight_layout()

    if save_plots:
        plt.savefig(f'{output_dir}/training_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {output_dir}/training_metrics_comparison.png")

    # plt.show()

    for metric, title in metrics.items():
        plot_single_metric(df, metric, title, save_plots=save_plots, output_dir=output_dir)
        # plt.show()


def main(log_file_paths):
    print("=== Training Log Analysis ===")
    print(f"Processing {len(log_file_paths)} log files...")

    df = parse_all_logs(log_file_paths)

    if df.empty:
        print("No data to analyze!")
        return

    print("\n=== Summary Statistics ===")
    print(df.groupby(['scale', 'gpus', 'fsdp']).agg({
        'avg_tokens_per_sec': 'mean',
        'avg_training_tokens_pct': 'mean',
        'avg_mfu_pct': 'mean',
        'avg_tflops': 'mean'
    }).round(2))

    print("\n=== Creating Visualizations ===")
    create_visualizations(df)

    return df


if __name__ == "__main__":
    log_files = [
        # no fsdp, gpus=1, scale=2
        "logs/train_no_fsdp/lsai-460955.out",
        # no fsdp, gpus=1, scale=4
        "logs/train_no_fsdp/lsai-460958.out",
        # no fsdp, gpus=1, scale=6
        "logs/train_no_fsdp/lsai-460959.out",
        # no fsdp, gpus=1, scale=8
        "logs/train_no_fsdp/lsai-460964.out",
        # no fsdp, gpus=1, scale=10  (OOM)
        # "logs/train_no_fsdp/lsai-460969.out",
        # fsdp, gpus=2, scale=2
        "logs/train_fsdp/lsai-461379.out",
        # fsdp, gpus=2, scale=4
        "logs/train_fsdp/lsai-461380.out",
        # fsdp, gpus=2, scale=6
        "logs/train_fsdp/lsai-461406.out",
        # fsdp, gpus=2, scale=10
        "logs/train_fsdp/lsai-461407.out",
        # fsdp, gpus=2, scale=14
        "logs/train_fsdp/lsai-461412.out",
        # fsdp, gpus=16, scale=1
        "logs/train_fsdp/lsai-461073.out",
        # fsdp, gpus=16, scale=2
        "logs/train_fsdp/lsai-460981.out",
        # fsdp, gpus=16, scale=6
        "logs/train_fsdp/lsai-460997.out",
        # fsdp, gpus=16, scale=10,
        "logs/train_fsdp/lsai-461014.out",
        # fsdp, gpus=16, scale=14
        "logs/train_fsdp/lsai-461015.out"
    ]

    results_df = main(log_files)
    print(results_df)