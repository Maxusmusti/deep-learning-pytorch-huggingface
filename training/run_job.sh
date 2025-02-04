
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_DISABLE=1

HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file="./configs/accelerate_configs/deepspeed_zero3_multinode.yaml" \
    scripts/run_r1_grpo_phi.py --config receipes/grpo-granite.yaml