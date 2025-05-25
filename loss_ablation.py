import argparse
import os
from datetime import datetime

import re
import matplotlib.pyplot as plt
import statistics


# train a model with fsdp and without fsdp and compare loss via logs
def parse_log(filepath):
    step_data = {}

    with open(filepath, 'r') as file:
        for line in file:
            match = re.search(
                r"Step:\s*(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*Tokens per second:\s*([\d.]+)\s*\|\s*Training tokens per second\s*\(%\):\s*([\d.]+)\s*\|\s*MFU\s*\(%\):\s*([\d.]+)\s*\|\s*TFLOPs:\s*([\d.]+)", line
            )
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                tokens_per_sec = float(match.group(3))
                training_tokens_pct = float(match.group(4))
                mfu = float(match.group(5))
                tflops = float(match.group(6))

                step_data[step] = {
                    'loss': loss,
                    'tokens_per_sec': tokens_per_sec,
                    'training_tokens_pct': training_tokens_pct,
                    'mfu': mfu,
                    'tflops': tflops,
                }

    return step_data

def calculate_means(data):
    metrics = ['tokens_per_sec', 'training_tokens_pct', 'mfu', 'tflops']
    means = {key: statistics.mean([entry[key] for entry in data.values()]) for key in metrics}
    return means


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merger-logs",
        type=str,
    )
    parser.add_argument(
        "--default-logs",
        type=str,
    )
    return parser.parse_args()


def save_plot_with_timestamp(filename_prefix="loss_comparison", folder_name="plots", dpi=300):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{filename_prefix}_{timestamp}.png"
    output_path = os.path.join(output_folder, filename)

    # save the current matplotlib plot
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    print(f"Plot saved to {output_path}")
    return output_path
    

if __name__ == "__main__":
    args = get_args()

    # load both logs
    log_merger = parse_log(args.merger_logs)
    log_default = parse_log(args.default_logs)

    # get common steps
    common_steps = sorted(set(log_merger.keys()) & set(log_default.keys()))

    # plot Loss Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(common_steps, [log_merger[s]['loss'] for s in common_steps], label='Log FSDP x Flash Attention Loss', marker='o')
    plt.plot(common_steps, [log_default[s]['loss'] for s in common_steps], label='Log Default Loss', marker='x')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Comparison per Step')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot_with_timestamp()
    
    # compute differences in loss
    loss_diffs = {
        step: abs(log_merger[step]['loss'] - log_default[step]['loss']) for step in common_steps
    }

    # find the step with the maximum loss difference
    max_diff_step = max(loss_diffs, key=loss_diffs.get)
    loss1 = log_merger[max_diff_step]['loss']
    loss2 = log_default[max_diff_step]['loss']
    max_diff = loss_diffs[max_diff_step]

    print(f"\n=== Max Loss Difference ===")
    print(f"Step: {max_diff_step}")
    print(f"Log FSDP x Flash Attention Loss: {loss1:.4f}")
    print(f"Log Deafult Loss: {loss2:.4f}")
    print(f"Absolute Difference: {max_diff:.4f}")

    # print mean stats
    means_fsdp = calculate_means(log_merger)
    means_no_fsdp = calculate_means(log_default)

    print("\n=== Mean Metrics ===")
    print("Log FSDP x Flash Attention:")
    for k, v in means_fsdp.items():
        print(f"  {k}: {v:.2f}")

    print("Log Default:")
    for k, v in means_no_fsdp.items():
        print(f"  {k}: {v:.2f}")
        
        
  

