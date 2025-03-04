#!/bin/bash

# Helper script for running multiple gen_and_concat_top_embs.py successively with different slides
# Usage: /home/ftoelkes/code/lymphoma/data_preparation/embedding_generation/run_generate_uni_embeddings.sh [num_jobs] [slides_per_job] [start_slide] [mode] [patch_size]
# Change the underlying docker command depending on your use case

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
  SCREEN_NAME="generate_and_concat_embs_${MODE}_${i}"
  DOCKER_NAME="ftoelkes_run_${i}"
  GPU_DEVICE=$i
  JOB_START_SLIDE=$((START_SLIDE + i * SLIDES_PER_JOB))
  
  # Launch each job in a new screen session
  screen -dmS "$SCREEN_NAME" bash -c \
    "/home/ftoelkes/code/lymphoma/data_preparation/embedding_generation/run_gen_and_concat_top_embs.sh \
    $GPU_DEVICE $JOB_START_SLIDE $SLIDES_PER_JOB $MODE $PATCH_SIZE $DATASET ; exec bash"
done




# Usage for run_gen_and_concat_top_embs.sh:

# Example Usage of run_mutiple_gen_concat_top_embs.sh (five jobs are limit per node):

# /home/ftoelkes/code/lymphoma/data_preparation/embedding_generation/run_mutiple_gen_concat_top_embs.sh 1 0 11 train 1024um munich 

# screen -dmS generate_top_emb_train0 sh -c '/home/ftoelkes/code/lymphoma/data_preparation/embedding_generation/run_gen_and_concat_top_embs.sh 0 0 44 "train" "1024um" "kiel"; exec bash'

# screen -dmS generate_top_emb_train0 sh -c '/home/ftoelkes/code/lymphoma/data_preparation/embedding_generation/run_gen_and_concat_top_embs.sh 0 0 44 "test" "1024um" "kiel"; exec bash'



