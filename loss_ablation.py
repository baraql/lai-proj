import argparse

from utils import get_arg_parser, build_lr_scheduler, clip_grad_norm_, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype


# train a model with fsdp and without fsdp and compare loss via logs??
# or should we run it on the same node at once and compare there?..


def parse_training_logs(args):
    pass
    

# add another argument: scale the model 
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
    

if __name__ == "__main__":
  init_logger()
  args = get_args()
  parse_training_logs(args)
