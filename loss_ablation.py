import argparse
import os
from datetime import datetime

from utils import get_arg_parser, build_lr_scheduler, clip_grad_norm_, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype

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
        "--fsdp-logs",
        type=str,
    )
    parser.add_argument(
        "--no-fsdp-logs",
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
    init_logger()
    args = get_args()

    # load both logs
    log_fsdp = parse_log(args.fsdp_logs)
    log_no_fsdp = parse_log(args.no_fsdp_logs)

    # get common steps
    common_steps = sorted(set(log_fsdp.keys()) & set(log_no_fsdp.keys()))

    # plot Loss Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(common_steps, [log_fsdp[s]['loss'] for s in common_steps], label='Log FSDP Loss', marker='o')
    plt.plot(common_steps, [log_no_fsdp[s]['loss'] for s in common_steps], label='Log NO FSDP Loss', marker='x')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Comparison per Step')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot_with_timestamp()
    
    # compute differences in loss
    loss_diffs = {
        step: abs(log_fsdp[step]['loss'] - log_no_fsdp[step]['loss']) for step in common_steps
    }

    # find the step with the maximum loss difference
    max_diff_step = max(loss_diffs, key=loss_diffs.get)
    loss1 = log_fsdp[max_diff_step]['loss']
    loss2 = log_no_fsdp[max_diff_step]['loss']
    max_diff = loss_diffs[max_diff_step]

    print(f"\n=== Max Loss Difference ===")
    print(f"Step: {max_diff_step}")
    print(f"Log FSDP Loss: {loss1:.4f}")
    print(f"Log NO FSDP Loss: {loss2:.4f}")
    print(f"Absolute Difference: {max_diff:.4f}")

    # print mean stats
    means_fsdp = calculate_means(log_fsdp)
    means_no_fsdp = calculate_means(log_no_fsdp)

    print("\n=== Mean Metrics ===")
    print("Log FSDP:")
    for k, v in means_fsdp.items():
        print(f"  {k}: {v:.2f}")

    print("Log NO FSDP:")
    for k, v in means_no_fsdp.items():
        print(f"  {k}: {v:.2f}")
        
        
# === Max Loss Difference ===
# Step: 20
# Log FSDP Loss: 11.3200
# Log NO FSDP Loss: 11.3300
# Absolute Difference: 0.0100

# === Mean Metrics ===
# Log FSDP:
#   tokens_per_sec: 4814.07
#   training_tokens_pct: 27.98
#   mfu: 6.08
#   tflops: 60.09
# Log NO FSDP:
#   tokens_per_sec: 7515.06
#   training_tokens_pct: 23.94
#   mfu: 39.16
#   tflops: 387.34
  

