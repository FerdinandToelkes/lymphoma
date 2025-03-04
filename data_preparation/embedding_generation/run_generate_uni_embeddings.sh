#!/bin/bash

# Usage: /home/ftoelkes/code/lymphoma/data_preparation/embedding_generation/run_generate_uni_embeddings.sh [num_jobs] [slides_per_job] [start_slide] [mode] [patch_size] [dataset]
# Change the underlying screen command depending on your use case


# Input parameters with fallback to defaults
NUM_JOBS=${1:-5}                    # Default: 5 jobs
SLIDES_PER_JOB=${2:-44}             # Default: 44 slides per job
START_SLIDE=${3:-0}                 # Default: Start from slide 0
MODE=${4:-"train"}                  # Default: 'train'
PATCH_SIZE=${5:-"1024um"}           # Default: 1024um
DATASET=${6:-"kiel"}                # Default: kiel

# Validate mode to avoid invalid directory errors
if [ "$MODE" != "train" ] && [ "$MODE" != "test" ]; then
  echo "Error: Unsupported mode '$MODE'. Supported modes are 'train' and 'test'."
  exit 1
fi


# Main loop to create jobs
for ((i=0; i<NUM_JOBS; i++))
do
  SCREEN_NAME="generate_embeddings_${MODE}${i}"
  DOCKER_NAME="ftoelkes_run_${i}"
  GPU_DEVICE="device=${i}"
  JOB_START_SLIDE=$((START_SLIDE + i * SLIDES_PER_JOB))
  
  screen -dmS "$SCREEN_NAME" sh -c \
    "docker run --shm-size=400gb --gpus \"$GPU_DEVICE\" --name $DOCKER_NAME -it -u \`id -u $USER\` --rm \
    -v /home/ftoelkes/code/lymphoma:/mnt \
    -v /home/ftoelkes/preprocessed_data/${MODE}_data:/data \
    -v /home/ftoelkes/code/lymphoma/vit_large_patch16_256.dinov2.uni_mass100k:/model \
    ftoelkes_lymphoma python3 -m data_preparation.embedding_generation.generate_uni_embeddings \
    --patch_size=\"$PATCH_SIZE\" --dataset=\"$DATASET\" --target_size=2048 --save_attentions \
    --starting_slide=$JOB_START_SLIDE --nr_of_slides=$SLIDES_PER_JOB ; exec bash"
done

# --patch_size=\"$PATCH_SIZE\" --target_size=2048 --save_attentions \
# --starting_slide=$JOB_START_SLIDE --nr_of_slides=$SLIDES_PER_JOB --no_normalize; exec bash"

# Example usage:
# /home/ftoelkes/code/lymphoma/data_preparation/embedding_generation/run_generate_uni_embeddings.sh 1 6 0 test 1024um munich
# /home/ftoelkes/code/lymphoma/data_preparation/embedding_generation/run_generate_uni_embeddings.sh 8 6 48 test 1024um munich
# /home/ftoelkes/code/lymphoma/data_preparation/embedding_generation/run_generate_uni_embeddings.sh 8 6 96 test 1024um munich

