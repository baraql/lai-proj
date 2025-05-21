import os
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler


# patch torch.distributed.launcher to prevent crashing when result is None
import torch.distributed.launcher.api as _api
_orig_launch = _api.launch_agent
def _safe_launch(*args, **kwargs):
    result = _orig_launch(*args, **kwargs)
    if result is None:
        class _DummyResult:
            def is_failed(self): return False
        return _DummyResult()
    return result
_api.launch_agent = _safe_launch

from dataset import CollatorForCLM, ParquetDataset
from model import Transformer
from utils import (
    build_lr_scheduler, clip_grad_norm_, get_args, get_num_params,
    get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE,
    set_default_dtype, set_seed
)

def train(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        logger.info(f"Experiment args: {args}")
        logger.info(f"[rank {dist.get_rank()}] world size: {dist.get_world_size()}")
        logger.info("Initializing tokenizer and dataset...")

    # tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, args.batch_size * args.training_steps)
    train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=train_collator)
    train_dl_iterator = iter(train_dl)

    # model setup
    if local_rank == 0:
        logger.info("Building model...")
        
    model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]
    model_config = args.scaling_strategy.scale(args.scaling_factor)
    model_config.vocab_size = tokenizer.vocab_size
    
    if local_rank == 0:
        logger.info(f"Loading a model with scale={args.scaling_factor}, scaling_strategy={args.scaling_strategy}")
        logger.info(f"Model config: {model_config}")
    
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

    model.train()

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer)
    lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)

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
        
        # Important: set epoch for sampler
        train_sampler.set_epoch(train_step)
        
        train_step += 1

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
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1).float(), 
            labels.flatten(0, 1), 
            reduction="sum"
        )
        loss = loss / num_items_in_batch
        del logits
        loss.backward()

        clip_grad_norm_(model.parameters(), args.grad_max_norm)

        optimizer.step()
        lr_scheduler.step()

        # logging
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

        if args.profile and args.profile_step_end == train_step:
            torch.cuda.cudart().cudaProfilerStop()

    if local_rank == 0:
        logger.info("Training complete")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    init_logger()
    args = get_args()
    if args.set_seed is not None:
        set_seed(args.set_seed)
        logger.info(f"Set seed: {args.set_seed}")
    train(args)
