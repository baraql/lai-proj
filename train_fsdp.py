import torch.distributed.launcher.api as _api
_orig_launch = _api.launch_agent
def _safe_launch(*args, **kwargs):
    result = _orig_launch(*args, **kwargs)
    if result is None:
        # dummy object with is_failed() → False
        class _R: 
            def is_failed(self): 
                return False
        return _R()
    return result
_api.launch_agent = _safe_launch

import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import CollatorForCLM, ParquetDataset
from model import Transformer, TransformerModelArgs, scale_model_config
from utils import build_lr_scheduler, clip_grad_norm_, get_args, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype, set_seed

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

def train(args):
  local_rank = int(os.environ["LOCAL_RANK"])
  if local_rank == 0:
    logger.info(f"Experiment args: {args}")
  # Init
  device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
  torch.cuda.set_device(local_rank)
  model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]

  dist.init_process_group(backend="nccl", init_method="env://")

  # Set up DataLoader
  if local_rank == 0:
    logger.info(f"[rank {dist.get_rank()}] world size: {dist.get_world_size()}")
    logger.info("Setting up DataLoaders...")
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
  train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, args.batch_size*args.training_steps)
  train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
  train_dl = DataLoader(train_ds,
                        batch_size=args.batch_size,
                        collate_fn=train_collator)
  train_dl_iterator = iter(train_dl)

  # Set up Model
  if local_rank == 0:
    logger.info("Setting up Model...")
  model_config = TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=tokenizer.vocab_size,
        seq_len=args.sequence_length,
    )
  
  if args.scale > 1:
    model_config = scale_model_config(model_config=model_config, scale=args.scale, scale_only_n_layers=True)
    
    
  with set_default_dtype(model_dtype):
    model = Transformer(model_config).to(device)
    total = sum(p.numel() for p in model.parameters())
    print("Total params:", total)

  model = FSDP(model)
  logger.info(f"[rank {dist.get_rank()}] model is now: {model.__class__.__name__}")
  local = sum(p.numel() for p in model.parameters())
  logger.info("[rank %d] local params: %d", dist.get_rank(), local)

    
  if args.compile:
    if local_rank == 0:
      logger.info("Using `torch.compile`")
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

  if local_rank == 0:
    logger.info("Starting training!")
    
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
        logger.info(f"Step: {train_step} | Loss: {loss.item():.2f} | Tokens per second: {tps:.2f} | Training tokens per second (%): {100*training_tps/tps:.2f} | MFU (%): {mfu:.2f} | TFLOPs: {tflops:.2f}")
      ntokens_since_last_log = 0
      ntraining_tokens_since_last_log = 0
      time_last_log = time.perf_counter()
    
    # Profiling
    if args.profile and args.profile_step_end == train_step:
      torch.cuda.cudart().cudaProfilerStop()


  if local_rank == 0:
    logger.info("Training completed")
    
  dist.barrier()  # Wait for all processes
  dist.destroy_process_group()
  

if __name__ == "__main__":
  init_logger()
  args = get_args()
  if args.set_seed is not None:
    set_seed(args.set_seed)
    logger.info(f"Setting seed to {args.set_seed}")
  train(args)
