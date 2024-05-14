#!/usr/bin/env bash
source ./train_setup.sh

: ${MAX_EPOCHS:=3}
: ${TRAIN_ITERS:=-1}
: ${VALID_ITERS:=1.0}
: ${INIT_METHOD_STD:=0.02}
: ${LAYERNORM_EPSILON:=1e-6}
: ${WARMUP_STEPS:=10}

: ${SEQ_LENGTH:=2048}
: ${HS:=4096}
: ${TP:=8}
: ${PP:=8}
: ${N_LAYERS:=32}
: ${N_AH:=32}
: ${UBS:=1}
: ${FFN_HS:=14336}
: ${GBS:=32}
: ${KV_HEADS=8}
: ${KV_CHANNELS=128}
: ${NUM_EXPERTS=8}
: ${MOE_TOPK=2}
: ${MOE_COEFF=0.02}
: ${ROPE_THETA=1000000.0}
: ${VOCAB_SIZE=32000}
: ${LR=3e-4}
: ${MIN_LR=3e-5}
: ${BETAS=[0.9,0.95]}
: ${WD=0.1}

: ${DATASET_PATH=$HOME/examples_datasets}
: ${EXP_DIR=./nemo_experiments}

: ${LIMIT_VAL_BATCHES=0}
: ${SAVE_TOP_K=5}
: ${CHECKPOINT_LOAD=model.use_cpu_initialization=True}

echo "SEQ_LEN=$SEQ_LENGTH, HS=$HS, FFN_HS=$FFN_HS TP=$TP PP=$PP N_LAYERS=$N_LAYERS N_AH=$N_AH GBS=$GBS UBS=$UBS TRAIN_ITERS=$TRAIN_ITERS"

LOG_PATH=logs/$SLURM_JOB_ID/$NODEID/
mkdir -p $LOG_PATH

$MAYBE_COMPILE torchrun $DISTRIBUTED_ARGS megatron_mixtral_pretraining.py  \
    --config-path=conf \
    --config-name=megatron_mixtral_config \
    trainer.devices=$PROCESSES_PER_NODE \
    trainer.num_nodes=$NTASKS \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.max_steps=$TRAIN_ITERS \
    trainer.val_check_interval=$VALID_ITERS \
    trainer.log_every_n_steps=1 \
    trainer.limit_val_batches=$LIMIT_VAL_BATCHES \
    trainer.limit_test_batches=1 \
    trainer.accumulate_grad_batches=1 \
    trainer.precision=32 \
    model.megatron_amp_O2=$megatron_amp_O2 \
    +trainer.num_sanity_val_steps=0 \
    model.tokenizer.vocab_size=$VOCAB_SIZE \
    model.micro_batch_size=$UBS \
    model.global_batch_size=$GBS \
    model.tensor_model_parallel_size=$TP \
    model.pipeline_model_parallel_size=$PP \
    model.max_position_embeddings=$SEQ_LENGTH \
    model.encoder_seq_length=$SEQ_LENGTH \
    model.hidden_size=$HS \
    model.ffn_hidden_size=$FFN_HS \
    model.num_layers=$N_LAYERS \
    model.num_attention_heads=$N_AH \
    model.init_method_std=$INIT_METHOD_STD \
    model.hidden_dropout=0 \
    model.num_kv_heads=$KV_HEADS \
    model.kv_channels=$KV_CHANNELS \
    model.layernorm_epsilon=$LAYERNORM_EPSILON \
    model.num_moe_experts=$NUM_EXPERTS \
    model.num_experts_per_tok=$MOE_TOPK \
    model.moe_aux_loss_coeff=$MOE_COEFF \
    model.rope_theta=$ROPE_THETA \
    model.data.hf_dataset_path=$DATASET_PATH \
    model.data.num_workers=1 \
    model.data.seq_length=$SEQ_LENGTH \
    model.optim.name=$OPTIM_NAME \
    model.optim.lr=$LR \
    model.optim.betas=$BETAS \
    model.optim.weight_decay=$WD \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=$WARMUP_STEPS \
    model.optim.sched.constant_steps=0 \
    model.optim.sched.min_lr=$MIN_LR \
    model.optim.capturable=True \
    model.activations_checkpoint_granularity=full \
    model.activations_checkpoint_method=uniform \
    model.activations_checkpoint_num_layers=1 \
    +model.save_xser=True \
    exp_manager.exp_dir=$EXP_DIR \
    exp_manager.checkpoint_callback_params.save_top_k=$SAVE_TOP_K \
    exp_manager.checkpoint_callback_params.every_n_epochs=1 \
    exp_manager.create_tensorboard_logger=$CREATE_TB_LOGGER \
    exp_manager.resume_if_exists=False \
    exp_manager.resume_ignore_no_checkpoint=False \
    exp_manager.create_checkpoint_callback=$CHECKPOINT_CALLBACK \
    exp_manager.checkpoint_callback_params.save_last=True \
    exp_manager.checkpoint_callback_params.save_weights_only=True \
    $CHECKPOINT_LOAD   2>&1  | tee  $LOG_PATH/log

# Note: to resume training using a checkpoint, please add the following configuration above, adjusting for your checkpoint path
    # model.use_cpu_initialization=False \
    # +model.load_xser=True \
    # +model.resume_from_checkpoint='/root/scripts/example_datasets/llamav2_weights/llama7b_hf_converted_nemo_v3//mp_rank_07/model_optim_rng.ckpt' \
# To use mixed precision optimizer, add
    # model.megatron_amp_O2=True \
