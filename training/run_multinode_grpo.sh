

# assign args to variables
config_file=$1
output_dir=$2
main_process_ip=$3
machine_rank=$4


export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_DISABLE=1

HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file="./configs/accelerate_configs/deepspeed_zero3_multinode.yaml" \
    --machine_rank $machine_rank \
    --main_process_ip $main_process_ip \
    scripts/run_r1_grpo_length_wait.py --config $config_file \
        --output_dir $output_dir \
