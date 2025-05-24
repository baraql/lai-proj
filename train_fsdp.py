import os
import time
import functools

import psutil 
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
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


from dataset import CollatorForCLM, ParquetDataset
from model import Transformer, ScalingStrategy, TransformerBlock
from utils import build_lr_scheduler, clip_grad_norm_, get_args, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype, set_seed


# Only the node with the global rank 0 will print it 
def log_dist(message):
  if int(os.environ["RANK"]) == 0:
      logger.info(f"[RANK 0 / {int(os.environ["WORLD_SIZE"])}] {message}")


def print_time_stats(start_time, end_time):
  elapsed = end_time - start_time
  minutes = int(elapsed // 60)
  seconds = int(elapsed % 60)
  log_dist(f"Took {minutes} min {seconds} sec")
    
    
def print_GPU_stats():
  log_dist(f"AVAILABLE GPUS: {int(os.environ["WORLD_SIZE"])}")
  log_dist(f"NODES: {int(os.environ["WORLD_SIZE"]) / int(os.environ["LOCAL_WORLD_SIZE"])}")
  
  
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
    
    
def train(args):
  print_GPU_stats()
  print_memory_info()
  
  start_time = time.time()

  local_rank = int(os.environ["LOCAL_RANK"])

  log_dist(f"Experiment args: {args}")
  
  # SET UP DATALOADER
  log_dist(f"world size: {int(os.environ["WORLD_SIZE"])}")
  log_dist("Setting up DataLoaders...")
    
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
  train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, args.batch_size*args.training_steps)
  train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
  train_dl = DataLoader(train_ds,
                        batch_size=args.batch_size,
                        collate_fn=train_collator)
  train_dl_iterator = iter(train_dl)

  # PREPARE MODEL CONFIG
  log_dist("Setting up Model...")
  model_config = args.scaling_strategy.scale(args.scaling_factor)
  model_config.vocab_size = tokenizer.vocab_size
  log_dist(f"Loading a model with scale={args.scaling_factor}, scaling_strategy={args.scaling_strategy}, config:\n{model_config}")


  # INIT THE MODEL 
  model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]
  device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
  
  with set_default_dtype(model_dtype):
    ## !! don't call .to(device) here, it should be loaded to RAM first !!
    model = Transformer(model_config)
  
  total_params = sum(p.numel() for p in model.parameters())
  print("Total model parameters:", total_params)
  
  # SETTING CUDA DEVICE
  local_rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local_rank)
  
  
  # SETUP TORCH DISTRIBUTED 
  if not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="env://")
        

  # FSDP 
  # we need to specify the wrap policy otw the default policy will wrap whole layers at once and they still don't fit
  auto_wrap_policy = functools.partial(
      transformer_auto_wrap_policy,
      transformer_layer_cls={
          TransformerBlock,
      },
  )

  # configure mixed precision for FSDP
  mixed_precision_policy = MixedPrecision(
      param_dtype=model_dtype,
      reduce_dtype=model_dtype,
      buffer_dtype=model_dtype
  )


  # apply FSDP directly to the model
  log_dist("Wrapping model with FSDP")
  model = FSDP(
      model,
      auto_wrap_policy=auto_wrap_policy,
      mixed_precision=mixed_precision_policy,
      sharding_strategy=ShardingStrategy.FULL_SHARD,  # most memory efficient
      device_id=torch.cuda.current_device(), ## !! important to specify it here !!
      cpu_offload=CPUOffload(offload_params=True),  # offload parameters to CPU when not in use
      limit_all_gathers=True,  # prevent OOM during all-gathers
  )
    
  log_dist(f"The model is now: {model.__class__.__name__}")
  local_params = sum(p.numel() for p in model.parameters())
  logger.info("[rank %d] local params: %d", dist.get_rank(), local_params)
    
  if args.compile:
    log_dist("Using `torch.compile`")
    model = torch.compile(model, fullgraph=True)
  
  model.train() # turn on training mode

  # Build Optimizers & LR Scheduler
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer)
  lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)

  # Utils
  num_flop_per_token = get_num_flop_per_token(
      get_num_params(model, exclude_embedding=True),
      model_config,
  )

  ntokens_since_last_log = 0
  ntraining_tokens_since_last_log = 0
  time_last_log = time.perf_counter()

  log_dist("Starting training!")
    
  train_step = 0
  
  while train_step < args.training_steps:
    train_step += 1

    # Profiling
    if args.profile and args.profile_step_start == train_step:
      torch.cuda.cudart().cudaProfilerStart()
      torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

    input_ids, labels = next(train_dl_iterator)
    ntokens_since_last_log += args.batch_size * args.sequence_length
    num_items_in_batch = labels.ne(-100).sum()
    ntraining_tokens_since_last_log += num_items_in_batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum")
    loss = loss / num_items_in_batch
    del logits
    loss.backward()

    # Clip gradients
    clip_grad_norm_(model.parameters(), args.grad_max_norm)

    optimizer.step()
    lr_scheduler.step()

    # Logging
    if (train_step == 1 or train_step % args.logging_frequency == 0):
      time_delta = time.perf_counter() - time_last_log
      # tokens per second per device, abbreviated as tps
      tps = ntokens_since_last_log / time_delta 
      mfu = 100 * num_flop_per_token * tps / 989e12
      tflops = num_flop_per_token * tps / 1e12
      training_tps = ntraining_tokens_since_last_log / time_delta

      if local_rank == 0:
        log_dist(f"Step: {train_step} | Loss: {loss.item():.2f} | Tokens per second: {tps:.2f} | Training tokens per second (%): {100*training_tps/tps:.2f} | MFU (%): {mfu:.2f} | TFLOPs: {tflops:.2f}")
      ntokens_since_last_log = 0
      ntraining_tokens_since_last_log = 0
      time_last_log = time.perf_counter()
    
    # Profiling
    if args.profile and args.profile_step_end == train_step:
      torch.cuda.cudart().cudaProfilerStop()

  log_dist("Training completed")
    
  end_time = time.time()
  print_time_stats(start_time, end_time)
  
  dist.barrier()  # Wait for all processes
  dist.destroy_process_group()
  

if __name__ == "__main__":
  init_logger()
  args = get_args()
  if args.set_seed is not None:
    set_seed(args.set_seed)
    log_dist(f"Setting seed to {args.set_seed}")
  train(args)
