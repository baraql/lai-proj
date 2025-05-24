import gc 
import os
import time
import functools
import psutil

import torch 
from transformers import AutoTokenizer

from model import Transformer, ScalingStrategy, TransformerBlock
from utils import PRECISION_STR_TO_DTYPE, set_default_dtype, init_logger, logger

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
import torch.distributed as dist



def log_dist(message):
    if int(os.environ["RANK"]) == 0:
        logger.info(f"[RANK 0 / {int(os.environ["WORLD_SIZE"])}] {message}")
        

def cleanup():
    if dist.is_initialized():
        # dist.barrier()
        dist.destroy_process_group()

    gc.collect()              # python garbage collection
    torch.cuda.empty_cache()  # clear unused cached memory
    torch.cuda.ipc_collect()  # clear memory between processes (optional but good)


def print_time_stats(start_time, end_time):
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    log_dist(f"Took {minutes} min {seconds} sec")
    
    
def get_RAM_gb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss 
    return mem_bytes / (1024 ** 3)
    
    
# default version will load the model with dim=4096, n_layers=32
def load_model_fsdp(scaling_factor: int = 4, scaling_strategy: ScalingStrategy = ScalingStrategy.ALL):    
    
    start_time = time.time()
    try:
  
        # INIT MODEL CONFIG
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")  
        model_config = scaling_strategy.scale(scaling_factor=scaling_factor)
        model_config.vocab_size = tokenizer.vocab_size
        
        log_dist(f"Loading a model with scale={scaling_factor}, scaling_strategy={scaling_strategy}, config:\n{model_config}")

        does_fit = None
        model_dtype = PRECISION_STR_TO_DTYPE["bf16"]
        
        device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
        
        # useless if we're loading the model to RAM
        # torch.cuda.reset_peak_memory_stats()
        # log_dist(f"Reset memory peak")
        # before_alloc = torch.cuda.memory_allocated(device)
        
        before_alloc = get_RAM_gb()

        # INIT THE MODEL
        with set_default_dtype(model_dtype):
            # should init on CPU so should fit!?? i believe so
            model = Transformer(model_config)
        
        # after_alloc = torch.cuda.memory_allocated(device)
        # peak_alloc = torch.cuda.max_memory_allocated(device)
        after_alloc = get_RAM_gb()

        log_dist(f"Memory used by model: {(after_alloc - before_alloc):.2f} GB")
        # print(f"Peak memory allocated during model init: {peak_alloc / (1024 ** 3):.2f} MB")
            
        # log parameter count before wrapping
        total_params = sum(p.numel() for p in model.parameters())
        log_dist(f"Total model parameters: {total_params:,}")

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)


        # SETUP
        if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
                
        # # FSDP params 
        # auto_wrap_policy = functools.partial(
        #     transformer_auto_wrap_policy,
        #     transformer_layer_cls={
        #         TransformerBlock,
        #     },
        # )

        
        # # configure mixed precision for FSDP
        # mixed_precision_policy = MixedPrecision(
        #     param_dtype=model_dtype,
        #     reduce_dtype=model_dtype,
        #     buffer_dtype=model_dtype
        # )
        

        # # apply FSDP directly to the model
        # log_dist("Wrapping model with FSDP")
        # model = FSDP(
        #     model,
        #     auto_wrap_policy=auto_wrap_policy,
        #     mixed_precision=mixed_precision_policy,
        #     sharding_strategy=ShardingStrategy.FULL_SHARD,  # most memory efficient
        #     device_id=torch.cuda.current_device(),
        #     cpu_offload=CPUOffload(offload_params=True),  # offload parameters to CPU when not in use
        #     limit_all_gathers=True,  # prevent OOM during all-gathers
        # )
                        
        # # after wrapping with FSDP
        # log_dist(f"[rank {dist.get_rank()}] model is now: {model.__class__.__name__}")
        # local_params = sum(p.numel() for p in model.parameters())
        # log_dist(f"[rank {dist.get_rank()}] local params: {local_params}")
        
        # # Test with a small forward pass
        # batch_size = 1
        # input_ids = torch.ones(batch_size, model.config.seq_len, dtype=torch.long, device=device)
        
        # # optional: run a small forward pass to verify memory usage
        # with torch.no_grad():
        #     try:
        #         outputs = model(input_ids)
        #         log_dist(f"Forward pass succeeded with shape: {outputs.shape}")
        #         does_fit = True
        #     except RuntimeError as e:
        #         log_dist(f"Forward pass failed: {e}")
        #         does_fit = False  
                        
    except RuntimeError as e:
        log_dist(f"Error while loading the model!")
        log_dist(e)
        does_fit = False
    
    end_time = time.time()
    print_time_stats(start_time, end_time)
    log_dist("\n\n")
    
    cleanup()
    
    return does_fit
        
    

# running it with a precision threshold (doesn't really work lol bc we need the model parmams to be devisible by certain numbers + they cannot be float)
# so keep precision = 0 for now 
def binary_search(low: int, high: int, scaling_strategy: ScalingStrategy):
    precision: int = 0
    log_dist(f"Running binary search with scale low={low}, high={high}, precision={precision}, scaling_strategy={scaling_strategy}")
    assert low < high 
    
    best_fit = None
    
    while low <= high:
        mid = (low + high) // 2
        
        # mid = round(mid, precision)
        
        # trying to load the model with `scale = mid`
        does_fit = load_model_fsdp(scaling_factor=mid, scaling_strategy=scaling_strategy)
        
        if does_fit:
            best_fit = mid  # model fits, update best_fit
            low = mid + pow(10, -precision)   # try more layers
            
        else:
            # model doesn't fit, try fewer layers
            high = mid - pow(10, -precision)

        cleanup()
        
    return best_fit


def print_memory_info():
    mem = psutil.virtual_memory()
    log_dist(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    log_dist(f"Available RAM: {mem.available / (1024**3):.2f} GB")
    tasks = int(os.environ["LOCAL_WORLD_SIZE"])
    per_process_mem = mem.available / ((1024**3) * tasks)
    log_dist(f"Available per-process RAM: {per_process_mem:.2f} GB")
    
    if torch.cuda.is_available():
        i = 0
        log_dist(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        log_dist(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
        log_dist(f"  Allocated memory: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
        log_dist(f"  Cached memory: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
    else:
        log_dist("No GPU available")




if __name__ == "__main__":
    init_logger()
    # log_dist(f"Starting the main function")

    # todo: move out to args? 
    scaling_strategy = ScalingStrategy.ALL
    
    torch.manual_seed(42)
    print_memory_info()
    
    
    # if scaling_strategy == ScalingStrategy.ALL: 
    # # best fit = ?? (see logs TODO)
    # # number of params ??
    #     low = 15
    #     high = 19
    # else:
    # # best fit = ?? (see logs TODO)
    # # number of params ??
    #     low = 20
    #     high = 24
    
    # Running with 4 processes: 
    # best_fit = binary_search(low=19, high=30, scaling_strategy=scaling_strategy)
    # log_dist(f"Best fit: {best_fit}")
    
    # load_model_fsdp(scaling_factor=19, scaling_strategy=scaling_strategy, auto_wrap_policy=auto_wrap_policy)
    # 
    
    # for running with 4 GPUS (memory ~193.81 GB per a process), best fit = ?? (with 22 already getting OOM), see logs/load_model_fsdp/lsai-457520.out
    # for running with 1 GPU (memory ~775.24 GB per a process), best fit = ??

    load_model_fsdp(scaling_factor=22, scaling_strategy=scaling_strategy)


    
