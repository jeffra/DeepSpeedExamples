
# SC21 Instructions

Goal: reproduce running a 50 billion parameter model with x16 V100s on [Cori](https://docs-dev.nersc.gov/cgpu/hardware/). In order to even start training this model size with this limited hardware budget we will utilize all of the features outlined in our SC21 paper "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning".

In this experiment we are not training a model end-to-end, we will just show that we are able to train 100 iterations.

The specific model architecture chosen here is outlined in Table 1 of our paper but is reproduced here:

Nodes | Parameter count | Hidden dimension | Number of layers | Batch size | Model parallelism | Parameter offload | Optimizer offload
----- | --------------- | ---------------- | ---------------- | ---------- | ----------------- | ----------------- | -----------------
1     | 50 billion      | 8192             | 62               | 4          | 1                 | CPU               | NVMe

Please refer to the following three files:

1) [Megatron-LM-v1.1.5-ZeRO3/examples/run_sc21.sh](https://github.com/jeffra/DeepSpeedExamples/blob/sc21i/Megatron-LM-v1.1.5-ZeRO3/examples/run_sc21.sh)
2) [Megatron-LM-v1.1.5-ZeRO3/examples/run_sc21_small_test.sh](https://github.com/jeffra/DeepSpeedExamples/blob/sc21i/Megatron-LM-v1.1.5-ZeRO3/examples/run_sc21_small_test.sh)
3) [Megatron-LM-v1.1.5-ZeRO3/examples/ds_zero3_sc21.json](https://github.com/jeffra/DeepSpeedExamples/blob/sc21i/Megatron-LM-v1.1.5-ZeRO3/examples/ds_zero3_sc21.json)

## Docker
* We've created a docker image that includes a small amount of training data that we use for performance testing.
* Image built and available on [docker hub](https://hub.docker.com/r/deepspeed/sc21) via: `docker pull deepspeed/sc21:latest`
* See [Dockerfile](https://github.com/jeffra/DeepSpeedExamples/blob/sc21i/sc21-docker/Dockerfile) for more details.
* The [SC21-DeepSpeed runtime](https://github.com/jeffra/deepspeed/tree/sc21i) is pre-installed in this docker image.

## Important Notes
* Please review the run script first, there are some required edits to ensure torch distributed works properly (e.g., set torch distributed master address).
* You'll want to launch the `run_sc21.sh` script passing in the appropriate node rank (i.e., 0 or 1). 
* In `examples/ds_zero3_sc21.json` you'll need to update the swap_tensor folder to your local NVMe path you have write access to, current it is set to `"/local_nvme"`. 
  * __This model size requires ~500 GB per node of available NVMe space to store temporary optimizer state during training.__

```json
{
  "swap_tensor": {
    "folder": "/local_nvme",
    ...
  },
  ...
}
```
