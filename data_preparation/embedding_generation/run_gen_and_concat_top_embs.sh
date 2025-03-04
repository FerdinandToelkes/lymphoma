#!/bin/bash

# Helper script for running multiple gen_and_concat_top_embs.py successively with different attended regions
# Usage: /home/ftoelkes/code/lymphoma/data_preparation/embedding_generation/run_gen_and_concat_top_embs.sh [device] [starting_slide] [nr_of_slides] [mode] [patch_size] [dataset]
# Change the underlying docker command depending on your use case

# Input arguments
DEVICE=${1:-0}                      # Default: 0
STARTING_SLIDE=${2:-0}              # Default: 0 
NR_OF_SLIDES=${3:-44}               # Default: 44 
MODE=${4:-"train"}                  # Default: "train" 
PATCH_SIZE=${5:-"1024um"}           # Default: "1024um" 
DATASET=${6:-"kiel"}                # Default: kiel

# Validate mode to avoid invalid directory errors
if [ "$MODE" != "train" ] && [ "$MODE" != "test" ]; then
  echo "Error: Unsupported mode '$MODE'. Supported modes are 'train' and 'test'."
  exit 1
fi


# Loop through attended regions
for NR_OF_ATTENDED_REGIONS in 1 5 10; do
    docker run --shm-size=400gb --gpus "device=$DEVICE" \
        --name "ftoelkes_run_${NR_OF_ATTENDED_REGIONS}_${DEVICE}" \
        -it -u `id -u $USER` --rm \
        -v /home/ftoelkes/code/lymphoma:/mnt \
        -v /home/ftoelkes/preprocessed_data/${MODE}_data:/data \
        -v /home/ftoelkes/code/lymphoma/vit_large_patch16_256.dinov2.uni_mass100k:/model \
        ftoelkes_lymphoma python3 -m data_preparation.embedding_generation.gen_and_concat_top_embs \
        --starting_slide="$STARTING_SLIDE" \
        --nr_of_slides="$NR_OF_SLIDES" \
        --nr_of_attended_regions="$NR_OF_ATTENDED_REGIONS" \
        --patch_size="$PATCH_SIZE" \
        --dataset="$DATASET"
done


# Example usage:

# screen -dmS generate_top_emb_train0 sh -c '/home/ftoelkes/code/lymphoma/data_preparation/embedding_generation/run_gen_and_concat_top_embs.sh 0 0 10 "train" "1024um" "munich"; exec bash'

# screen -dmS generate_top_emb_test0 sh -c '/home/ftoelkes/code/lymphoma/data_preparation/embedding_generation/run_gen_and_concat_top_embs.sh 0 0 10 "test" "1024um" "munich"; exec bash'



