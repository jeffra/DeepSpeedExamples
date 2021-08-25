#!/bin/bash

DATA_DIR=${HOME}/megatron-small-indexed/
DS_CONFIG=${HOME}/DS-SC21/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/examples/ds_zero3_sc21.json
BATCH_SIZE=2

deepspeed --num_nodes 1 --num_gpus 16 \
    pretrain_gpt2.py \
    --model-parallel-size 1 \
    --num-layers 62 \
    --hidden-size 8192 \
    --num-attention-heads 32 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --batch-size ${BATCH_SIZE} \
    --train-iters 320000 \
    --lr-decay-iters 320000 \
    --data-path ${DATA_DIR}/indexed/my-gpt2_text_document \
    --vocab-file ${DATA_DIR}/gpt2-vocab.json \
    --merge-file ${DATA_DIR}/gpt2-merges.txt \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 1.5e-4 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --warmup 0.01 \
    --checkpoint-activations \
    --log-interval 1 \
    --fp16 \
    --scattered-embeddings \
    --split-transformers \
    --deepspeed \
    --deepspeed_config ${DS_CONFIG} \
    --zero-stage 3 \
    --zero-reduce-bucket-size 50000000 \
    --zero-allgather-bucket-size 5000000000 \
    --cpu-optimizer \
    --zero-contigious-gradients \
    --zero-reduce-scatter \
    --deepspeed-activation-checkpointing \
    --checkpoint-num-layers 1 \
    --partition-activations \
    --checkpoint-in-cpu \
    --synchronize-each-layer \
    --contigious-checkpointing

