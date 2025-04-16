#!/bin/bash

exec &> >(tee -a output.log)

# Base directories
SPLITS_BASE="../configs/splits"
OUTPUT_BASE="../results/optimization/"

# Function to check if fold is completed
is_fold_completed() {
    local fold_dir="$1"
    # Check if trial_149 exists (meaning all 40 trials are done)
    if [ -d "$fold_dir/trial_149" ]; then
        return 0  # true in bash
    else
        return 1  # false in bash
    fi
}

# Iterate through all fold directories
for fold_dir in "$SPLITS_BASE"/fold_*; do
    # Extract the fold number from the directory name
    fold_num=$(basename "$fold_dir" | sed 's/fold_//')
    output_dir="$OUTPUT_BASE/fold_$fold_num"

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Skip if fold is completed
    if is_fold_completed "$output_dir"; then
        echo "Fold $fold_num: All trials completed (trial_149 exists). Skipping..."
        continue
    fi

    # Count existing trials by finding the highest trial number
    highest_trial=-1
    for trial_dir in "$output_dir"/trial_*; do
        if [ -d "$trial_dir" ]; then
            trial_num=$(basename "$trial_dir" | sed 's/trial_//')
            if [ "$trial_num" -gt "$highest_trial" ]; then
                highest_trial=$trial_num
            fi
        fi
    done

    # Calculate remaining trials
    remaining_trials=$((150 - (highest_trial + 1)))

    echo "Fold $fold_num: $((highest_trial + 1)) trials completed, running $remaining_trials more trials..."

    # Run optuna
    python optimize_hyperparams.py \
        --base-config ../configs/base_config.yml \
        --paths-config ../configs/paths.json \
        --splits-dir "$fold_dir" \
        --output-dir "$output_dir" \
        --n-trials $remaining_trials

    # Check if the run was successful
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing fold $fold_num"
        exit 1
    fi
done

echo "All folds processed successfully!"