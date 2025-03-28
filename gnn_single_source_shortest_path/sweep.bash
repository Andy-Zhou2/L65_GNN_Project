#!/bin/bash

# Define arrays for seed values and dataset sizes
seeds=(1 2 3 4)
num_graphs_values=(1000 2000 5000 10000 20000 40000 )

# Iterate over each seed
for seed in "${seeds[@]}"; do
  # Iterate over each dataset size
  for num_graphs in "${num_graphs_values[@]}"; do
    echo "Running with seed=${seed} and dataset.num_graphs=${num_graphs}"
    python -m gnn_single_source_shortest_path.gcn_main -m --config-name config seed=${seed} dataset.num_graphs=${num_graphs}
    # Check if the command was successful
    if [ $? -ne 0 ]; then
      echo "Command failed for seed=${seed} and dataset.num_graphs=${num_graphs}. Exiting."
      exit 1
    fi
  done
done
