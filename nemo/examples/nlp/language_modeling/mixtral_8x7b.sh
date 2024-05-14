#!/usr/bin/env bash

export SEQ_LENGTH=2048
export HS=4096
export TP=8
export PP=8
export N_LAYERS=32
export N_AH=32
export FFN_HS=14336
export GBS=32
export KV_HEADS=8
export KV_CHANNELS=128
export VOCAB_SIZE=32000
export NUM_EXPERTS=8
export MOE_TOPK=2
export MOE_COEFF=0.02
export ROPE_THETA=1000000.0

export MAX_EPOCHS=3

export INIT_METHOD_STD=0.02
export LAYERNORM_EPSILON=1e-6
export WARMUP_STEPS=10

export DATASET_PATH=./data/mixtral_sft
export EXP_DIR=./nemo_experiments

export LIMIT_VAL_BATCHES=1.0
export SAVE_TOP_K=5
export CHECKPOINT_LOAD="model.use_cpu_initialization=True"
# export CHECKPOINT_LOAD="model.use_cpu_initialization=False \
#     +model.load_xser=True \
#     +model.resume_from_checkpoint=./models/Mixtral-8x7B-v0.1-nemo-tp-8-pp-8/tp_rank_07_pp_rank_007/model_optim_rng.ckpt"

export LR=1e-5
export MIN_LR=1e-6

./test_mixtral.sh
