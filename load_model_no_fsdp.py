import gc 
import os
import time

import torch 
from transformers import AutoTokenizer
import argparse

from model import Transformer, TransformerModelArgs, scale_model_config, MIN_MODEL_CONFIG
from utils import PRECISION_STR_TO_DTYPE, set_default_dtype, init_logger, logger


def cleanup():
    gc.collect()              # python garbage collection
    torch.cuda.empty_cache()  # clear unused cached memory
    torch.cuda.ipc_collect()  # clear memory between processes (optional but good)


def print_time_stats(start_time, end_time):
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    logger.info(f"Took {minutes} min {seconds} sec")
    

    
# default version will load the model with dim=4096, n_layers=32
def load_model_no_fsdp(scale: int = 4, scale_only_n_layers: bool = True):
    start_time = time.time()
    device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
    model_dtype = PRECISION_STR_TO_DTYPE["bf16"]
  
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")


    if scale_only_n_layers:
        # scale dim, n_layers and n_heads 
        model_config = TransformerModelArgs(
            dim=4096,
            n_layers=8, # layers are multiple of 8
            n_heads=32,
            n_kv_heads=8,
            ffn_dim_multiplier=1.3,
            multiple_of=1024,
            rope_theta=500000,
            vocab_size=tokenizer.vocab_size,
            seq_len=4096,
        )
    else:
        model_config = MIN_MODEL_CONFIG
        
    model_config = scale_model_config(model_config, scale, scale_only_n_layers)
    
    logger.info(f"Loading a model with scale={scale}, dim={model_config.dim}, n_layers={model_config.n_layers}, n_heads={model_config.n_heads}")
   
    # exp_params_n = 2 * tokenizer.vocab_size * dim + n_layers * (12 * dim * dim + 2 * dim) + dim
    # logger.info(f"Expected model parameters: {exp_params_n}")
    
    # hidden_dim = 256 * ((int(8 * dim / 3) + 256 - 1) // 256)
    # total_params = 2 * tokenizer.vocab_size * dim + n_layers * (4*dim*dim + 3*dim*hidden_dim) + dim
    # logger.info(f"Another expected number of parameters: {total_params}")
    
    does_fit = None

    try:
        with set_default_dtype(model_dtype):
            model = Transformer(model_config).to(device)
        logger.info(f"Actual model parameters: {sum(p.numel() for p in model.parameters()):,}")
        does_fit = True
        
    except RuntimeError as e:
        logger.info(f"Error while loading the model!")
        logger.info(e)
        does_fit = False
    
    end_time = time.time()
    print_time_stats(start_time, end_time)
    logger.info("\n\n")
    
    cleanup()
    return does_fit
        
        
# not really used anymore
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_dim_scale",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_n_layers_scale",
        type=int,
        default=1,
    )
    return parser.parse_args()


# running it with a precision threshold (doesn't really work lol bc we need the model parmams to be devisible by certain numbers + they cannot be float)
# so keep precision = 0 for now 
def binary_search(low: int, high: int, precision: int = 0, scale_only_n_layers: bool = True):
    logger.info(f"Running binary search with scale low={low}, high={high}, precision={precision}, scale only n_layers={scale_only_n_layers}")
    assert low < high 
    
    best_fit = None
    
    while low <= high:
        mid = (low + high) // 2
        
        # mid = round(mid, precision)
        
        # trying to load the model with `scale = mid`
        does_fit = load_model_no_fsdp(scale=mid, scale_only_n_layers=scale_only_n_layers)
        
        if does_fit:
            best_fit = mid  # model fits, update best_fit
            low = mid + pow(10, -precision)   # try more layers
            
        else:
            # model doesn't fit, try fewer layers
            high = mid - pow(10, -precision)

        cleanup()
        
    return best_fit



# dim = 4096, n_layers = 4*8 => 8,053,329,920
# dim = 4096, n_layers = 8*8 => 15,032,913,920
# dim = 4096, n_layers = 12*8 => 22,012,497,920

# dim = 2*4096, n_layers = 32 => 30,065,303,552

# See logs at /lai-proj/logs/load_model_no_fsdp/lsai-446975.out
if __name__ == "__main__":
    init_logger()
    logger.info(f"Starting the main function")

    # todo: move out to args? 
    scale_only_n_layers = False
    
    # scaling only the number of layers, other params stay the same
    # found best fit scale = 27
    if scale_only_n_layers:
        low = 15
        high = 30
    
    # scale everything (starting with the minimum config)
    else:
        low = 1
        hight = 16
        
    best_fit = binary_search(low=low, high=high, scale_only_n_layers=scale_only_n_layers)
    logger.info(f"Best fit: {best_fit}")
    
    load_model_no_fsdp(scale=best_fit, scale_only_n_layers=scale_only_n_layers)
    
