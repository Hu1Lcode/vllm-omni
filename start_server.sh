export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export MEMORY_FRAGMENTATION=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export MULRI_STREAM_MEMORY_REUSE=1
# export MINDIE_SD_FA_TYPE=ascend_laser_attention

# export WAN_DUAL_BATCH_OVERLAP=1 # dbo
# export HCCL_OP_EXPANSION_MODE="AIV" # AIV
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

vllm serve /home/wjh/ltx2.3-diffusers \
    --omni \
    --model-class-name LTX23Pipeline \
    --port 8099 \
    --usp 8 \
    --ulysses-mode advanced_uaa \
    --use-hsdp \
    --hsdp-shard-size 8 \
    --vae-patch-parallel-size 8 \
    --vae-use-tiling \
    --enforce_eager \
    --log-stats 
    # --quantization int8 \
    # --profiler-config '{ "profiler": "torch","torch_profiler_dir": "./profiling"}'