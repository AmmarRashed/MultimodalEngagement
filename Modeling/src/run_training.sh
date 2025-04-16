#!/bin/bash

# Get the folder from command line argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <splits_folder_path>"
    exit 1
fi

SPLITS_DIR=$1
PATHS_CONFIG="../configs/paths.json"
#TRAINING_CONFIG="../configs/training_configs.yml"
TRAINING_CONFIG="../configs/ordinal_config.yml"

# Count number of CSV files in the splits directory
NUM_FOLDS=$(ls "$SPLITS_DIR"/*.csv | wc -l)

if [ $NUM_FOLDS -eq 0 ]; then
    echo "No CSV files found in $SPLITS_DIR"
    exit 1
fi

# Set number of parallel jobs (minimum of: num_folds, 10, num_cpu_cores)
NUM_CORES=$(nproc)
NUM_JOBS=$(( NUM_FOLDS < 4 ? NUM_FOLDS : 4 ))
NUM_JOBS=$(( NUM_JOBS < NUM_CORES ? NUM_JOBS : NUM_CORES ))

# Run the training
seq 0 $((NUM_FOLDS-1)) | parallel -j $NUM_JOBS python train_fold.py {} --paths-config $PATHS_CONFIG --training-config $TRAINING_CONFIG --splits-dir $SPLITS_DIR