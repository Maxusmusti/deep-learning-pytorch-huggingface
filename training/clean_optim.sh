#!/bin/bash
# Check if directory argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory=$1


# re-run the following every 20 mins
while true; do

    # Find all checkpoint directories and sort them by step number;
    # Store all global_step directories in an array
    global_step_dirs=()
    for checkpoint_dir in $(find "$directory" -maxdepth 1 -type d -name "checkpoint-*" | sort -V); do
        step_num=$(echo "$checkpoint_dir" | grep -o '[0-9]\+')
        
        # Find the global_step directory
        global_step_dir=$(find "$checkpoint_dir" -maxdepth 1 -type d -name "global_step*" | head -n 1)
        
        if [ -n "$global_step_dir" ]; then
            global_step_dirs+=("$global_step_dir")
        fi
    done

    # Calculate how many directories to delete (all except last 2)
    num_dirs=${#global_step_dirs[@]}
    dirs_to_delete=$((num_dirs - 2))


    # print all dirs_sto_delete
    echo "Global step dirs to delete: ${global_step_dirs[@]:0:$dirs_to_delete}"


    # Only proceed if we have more than 2 directories
    if [ $dirs_to_delete -gt 0 ]; then
        # print the global_step_dirs to delete
        echo "Global step dirs to delete: ${global_step_dirs[@]:0:$dirs_to_delete}"

        # Delete all but the last 2 global_step directories
        for ((i=0; i<dirs_to_delete; i++)); do
            rm -r "${global_step_dirs[i]}"
            echo "Deleting global_step directory: ${global_step_dirs[i]}"
        done
    else
        echo "Not enough directories to clean up (found $num_dirs, keeping 2)"
    fi

    echo "Cleanup complete"

    # Sleep for 20 minutes before next iteration
    sleep 1200
done

