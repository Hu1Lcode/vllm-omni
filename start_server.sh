export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export MEMORY_FRAGMENTATION=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export MULRI_STREAM_MEMORY_REUSE=1
# export MINDIE_SD_FA_TYPE=ascend_laser_attention

# export WAN_DUAL_BATCH_OVERLAP=1 # dbo
# export HCCL_OP_EXPANSION_MODE="AIV" # AIV
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve /home/wjh/ltx2.3-diffusers \
    --omni \
    --model-class-name LTX23Pipeline \
    --port 8099 \
    --use-hsdp \
    --hsdp-shard-size 4 \
    --vae-patch-parallel-size 4 \
    --vae-use-tiling \
    --enforce_eager \
    --log-stats 
    # --quantization int8 \
    # --profiler-config '{ "profiler": "torch","torch_profiler_dir": "./profiling"}'