import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).parent.absolute()
prefix = "MERGER"


def parse_log_file(file_path):
    with open(PROJECT_DIR / file_path, 'r') as f:
        content = f.read()

    scale_match = re.search(r'Loading a model with scale=(\d+)', content)

    if not scale_match:
        print(f"Warning: Could not extract scale configuration from {file_path}, skipping this file")
        return None

    scale = int(scale_match.group(1))
    
    flash_attention = 'flash_attention' in file_path

    if 'no_fsdp' in file_path:
        fsdp = False
        gpus = 1  # no_fsdp always uses 1 GPU
        nodes = 1 # and 1 node
    elif 'fsdp' in file_path:
        fsdp = True
        gpu_match = re.search(r'AVAILABLE GPUS: (\d+)', content)
        if not gpu_match:
            print(f"Warning: Could not extract GPUs configuration from FSDP file {file_path}, skipping this file")
            return None
        gpus = int(gpu_match.group(1))
        
        nodes_match = re.search(r'NODES:\s*([\d.]+)', content)
        if not nodes_match:
            print(f"Warning: Could not extract Nodes configuration from FSDP file {file_path}, skipping this file")
            return None
        # nodes_float = float(nodes_match.group(1))
        # nodes_int = int(nodes_float)
        nodes = int(float(nodes_match.group(1)))
    else:
        print(f"Warning: Could not determine gpus / nodes configuration from {file_path}, skipping this file")
        return None
    
    total_params_match = re.search(r'Total model parameters:\s*([\d,]+)', content)
    if not total_params_match:
        print(f"Warning: Could not extract total model params from log file {file_path}, skipping this file")
        return None
    total_params = int(total_params_match.group(1).replace(',', ''))
    


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
        'total_params': total_params,
        'scale': scale,
        'gpus': gpus,
        'nodes': nodes,
        'fsdp': fsdp,
        'flash_attention': flash_attention,
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
    print(df[['total_params', 'scale', 'gpus', 'nodes', 'fsdp', 'flash_attention', 'num_step_entries']].to_string(index=False))

    return df


def plot_single_metric(df, metric, title, ax=None, save_plots=False, output_dir=None, x_axis='scale'):
    assert x_axis in ['scale', 'total_params']
    
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
    df = df.sort_values(['fsdp', 'flash_attention', 'gpus', 'nodes'])

    # create unique configurations combining GPU / nodes count and FSDP
    df['config'] = df['gpus'].astype(str) + ' GPU' + (df['gpus'] > 1).map({True: 's', False: ''}) + ', ' + \
            df['nodes'].astype(str) + ' node' + (df['nodes'] > 1).map({True: 's', False: ''}) + \
            ' (' + df['fsdp'].map({True: 'FSDP', False: 'no FSDP'}) + ', ' + df['flash_attention'].map({True: 'FLASH ATTENTION', False: 'no FLASH ATTENTION'})+ ')'
    
    unique_configs = df['config'].drop_duplicates().tolist()

    # generate colors for each configuration
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_configs)))
    color_map = dict(zip(unique_configs, colors))

    # plot each configuration
    for config in unique_configs:
        config_data = df[df['config'] == config].sort_values(x_axis)  # sort by scale for proper line connection

        ax.scatter(config_data[x_axis], config_data[metric],
                   c=[color_map[config]],
                   label=config,
                   s=100, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)

        if len(config_data) > 1:
            ax.plot(config_data[x_axis], config_data[metric],
                    color=color_map[config], alpha=0.6, linewidth=2,
                    linestyle='-', marker='o', markersize=0, zorder=2)

    x_label = 'Model Scale' if x_axis == 'scale' else 'Model Parameters'
    ax.set_xlabel(x_label)
    ax.set_ylabel(title)
    ax.set_title(f'{title} vs {x_label}')
    ax.legend()
    ax.grid(True, alpha=0.3, zorder=1)

    if x_axis == 'scale':
        if all(df[x_axis] == df[x_axis].astype(int)):
            ax.set_xticks(sorted(df[x_axis].unique()))
    elif x_axis == 'total_params':
        def comma_formatter(x, pos):
            return f'{int(x):,}'
        ax.xaxis.set_major_formatter(FuncFormatter(comma_formatter))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


    if save_plots and output_dir and ax == plt.gca():  # Only save if this is a standalone plot
        safe_filename = metric.replace('_', '_').replace('%', 'pct')
        safe_filename += f"_{x_axis}"
        plt.savefig(f'{output_dir}/{prefix}_{safe_filename}.png', dpi=300, bbox_inches='tight')
        print(f"Individual plot saved to {output_dir}/{safe_filename}.png")

    return ax


def create_visualizations(df, save_plots=True, output_dir=PROJECT_DIR / 'plots', x_axis='scale'):
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
        plot_single_metric(df, metric, title, ax=axes[idx], x_axis=x_axis)

    plt.tight_layout()

    if save_plots:
        plt.savefig(f'{output_dir}/{prefix}_training_metrics_comparison_{x_axis}.png', dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {output_dir}/{prefix}_training_metrics_comparison_{x_axis}.png")

    # plt.show()

    for metric, title in metrics.items():
        plot_single_metric(df, metric, title, save_plots=save_plots, output_dir=output_dir, x_axis=x_axis)
        # plt.show()


def main(log_file_paths, x_axis):
    print("=== Training Log Analysis ===")
    print(f"Processing {len(log_file_paths)} log files...")

    df = parse_all_logs(log_file_paths)

    if df.empty:
        print("No data to analyze!")
        return

    print("\n=== Summary Statistics ===")
    print(df.groupby(['total_params', 'scale', 'gpus', 'fsdp', 'flash_attention']).agg({
        'avg_tokens_per_sec': 'mean',
        'avg_training_tokens_pct': 'mean',
        'avg_mfu_pct': 'mean',
        'avg_tflops': 'mean'
    }).round(2))

    print("\n=== Creating Visualizations ===")
    create_visualizations(df, x_axis=x_axis)

    return df


if __name__ == "__main__":
    
    log_files = [
    # fsdp, gpus=2, nodes=1, scale=2
    "logs/train_fsdp/lsai-466053.out",
    # fsdp, gpus=2, nodes=1, scale=4
    "logs/train_fsdp/lsai-466061.out",
    # fsdp, gpus=2, nodes=1, scale=6
    "logs/train_fsdp/lsai-466064.out",
    # fsdp, gpus=2, nodes=1, scale=10
    "logs/train_fsdp/lsai-466075.out",
    # fsdp, gpus=2, nodes=1, scale=14
    "logs/train_fsdp/lsai-466085.out",
    # fsdp, gpus=2, nodes=1, scale=15 OOM 
    # "logs/train_fsdp/lsai-466097.out",
    'logs/train_flash_attention_fsdp/lsai_scale1.out',
    'logs/train_flash_attention_fsdp/lsai_scale10.out',
    'logs/train_flash_attention_fsdp/lsai_scale14.out',
    'logs/train_flash_attention_fsdp/lsai_scale2.out',
    'logs/train_flash_attention_fsdp/lsai_scale4.out',
    'logs/train_flash_attention_fsdp/lsai_scale6.out',
    'logs/train_flash_attention_fsdp/lsai_scale8.out'
]

# change x_scale='total_params' to have total model parameters on X axis
results_df = main(log_files, x_axis='total_params')
print(results_df)