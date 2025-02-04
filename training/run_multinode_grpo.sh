
# args:
# 1: machine_rank
# 2: num_machines
# 3: num_processes
# 4: main_process_ip
# 5: main_process_port
# 6: config_file

# accept those required args, and raise error if not provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ]; then
    echo "Usage: $0 <machine_rank> <num_machines> <num_processes> <main_process_ip> <main_process_port> <config_file>"
    exit 1
fi

# assign args to variables
machine_rank=$1
num_machines=$2
num_processes=$3
main_process_ip=$4
main_process_port=$5
config_file=$6

export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_DISABLE=1

HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file="./configs/accelerate_configs/deepspeed_zero3_multinode.yaml" \
    --machine_rank $machine_rank \
    --num_machines $num_machines \
    --num_processes $num_processes \
    --main_process_ip $main_process_ip \
    --main_process_port $main_process_port \
    scripts/run_r1_grpo_phi.py --config $config_file

