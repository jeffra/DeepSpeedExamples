
# SC21 Instructions

Goal: reproduce running a 50 billion parameter model with x16 V100s on [Cori](https://docs-dev.nersc.gov/cgpu/hardware/). In order to even start training this model size with this limited hardware budget we will utilize all of the features outlined in our SC21 paper "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning".

In this experiment we are not training a model end-to-end, we will just show that we are able to train 100 iterations.

The specific model architecture chosen here is outlined in Table 1 of our paper but is reproduced here:

Nodes | Parameter count | Hidden dimension | Number of layers | Batch size | Model parallelism | Parameter offload | Optimizer offload
----- | --------------- | ---------------- | ---------------- | ---------- | ----------------- | ----------------- | -----------------
1     | 50 billion      | 8192             | 62               | 4          | 1                 | CPU               | NVMe

Please refer to the following two files:

1) `Megatron-LM-v1.1.5-ZeRO3/examples/run_sc21.sh`
2) `Megatron-LM-v1.1.5-ZeRO3/examples/ds_zero3_sc21.json`

## Important Notes
* Please review the run script first, there are some required edits to ensure torch distributed works properly (e.g., set torch distributed master address).
* You'll want to launch the `run_sc21.sh` script passing in the appropriate node rank (i.e., 0 or 1). 

