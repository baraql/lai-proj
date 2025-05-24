
to enable fused attention, simply add `--fused-attention` to the train args.
There were various options to play around with in assignment 2, but for example `--compile` doesn't work.
I suggest we use `--fused-optimizer` and `--sequence-length 2048` and then see what happens when we activate/deactivate `--fused-attention`. 
Currently train.py is logging the average mfu etc. at the end, if we want to plot the train loss we'd need to create a script that parses the logs.

https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.DotProductAttention
To enable fused attention we just set environment variables
NVTE_FUSED_ATTN=1
NVTE_FLASH_ATTN=0