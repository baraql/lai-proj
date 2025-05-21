import gc 
import os
import time

import torch 
from transformers import AutoTokenizer

from model import Transformer, ScalingStrategy
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
def load_model_no_fsdp(scaling_factor: int = 4, scaling_strategy: ScalingStrategy = ScalingStrategy.ALL):
    start_time = time.time()
    device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
    model_dtype = PRECISION_STR_TO_DTYPE["bf16"]
  
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")  
    model_config = scaling_strategy.scale(scaling_factor=scaling_factor)
    model_config.vocab_size = tokenizer.vocab_size
    
    logger.info(f"Loading a model with scale={scaling_factor}, scaling_strategy={scaling_strategy}, config:\n{model_config}")
   
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
        
    

# running it with a precision threshold (doesn't really work lol bc we need the model parmams to be devisible by certain numbers + they cannot be float)
# so keep precision = 0 for now 
def binary_search(low: int, high: int, scaling_strategy: ScalingStrategy):
    precision: int = 0
    logger.info(f"Running binary search with scale low={low}, high={high}, precision={precision}, scaling_strategy={scaling_strategy}")
    assert low < high 
    
    best_fit = None
    
    while low <= high:
        mid = (low + high) // 2
        
        # mid = round(mid, precision)
        
        # trying to load the model with `scale = mid`
        does_fit = load_model_no_fsdp(scaling_factor=mid, scaling_strategy=scaling_strategy)
        
        if does_fit:
            best_fit = mid  # model fits, update best_fit
            low = mid + pow(10, -precision)   # try more layers
            
        else:
            # model doesn't fit, try fewer layers
            high = mid - pow(10, -precision)

        cleanup()
        
    return best_fit


if __name__ == "__main__":
    init_logger()
    logger.info(f"Starting the main function")

    # todo: move out to args? 
    scaling_strategy = ScalingStrategy.N_LAYERS
    
    if scaling_strategy == ScalingStrategy.ALL: 
    # best fit = 19 (see logs logs/load_model_no_fsdp/lsai-453992.out)
    # number of params 46,322,328,320
        low = 1
        high = 20
    else:
    # best fit = 24 (see logs logs/load_model_no_fsdp/lsai-454054.out)
    # number of params 48,185,937,920
        low = 15
        high = 30
                
    best_fit = binary_search(low=low, high=high, scaling_strategy=scaling_strategy)
    logger.info(f"Best fit: {best_fit}")
    
    load_model_no_fsdp(scaling_factor=best_fit, scaling_strategy=scaling_strategy)
    
