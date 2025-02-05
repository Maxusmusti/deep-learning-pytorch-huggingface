

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




SESSION_NAME="deletion"

# Create new tmux session (or attach if exists) and run cleanup
tmux new-session -d -s $SESSION_NAME 2>/dev/null || true

# Send commands to the tmux session
tmux send-keys -t $SESSION_NAME "bash clean_optim.sh $output_dir" C-m
