#!/bin/bash

# change the screen command depending on which data set you want to sample (see below for single examples of command without this script) 
# Usage: /home/ftoelkes/code/lymphoma/resnet_ddp/run_multiple_trainings.sh [patch_size] [dataset] [data_dir] [data_specifier] [processes]
# Example: 
#screen -dmS train_resnets sh -c '/home/ftoelkes/code/lymphoma/resnet_ddp/run_multiple_trainings.sh 1024um kiel embeddings_dir patches 8; exec bash'

#screen -dmS train_resnets sh -c '/home/ftoelkes/code/lymphoma/resnet_ddp/run_multiple_trainings.sh 1024um kiel embeddings_dir_not_normalized patches 8; exec bash'





# Input parameters with fallback to defaults
PATCH_SIZE=${1:-"1024um"}         # Default to 1024um
DATASET=${2:-"kiel"}              # Default to 'kiel', can be 'kiel', 'swiss_1', 'swiss_2', 'multiply', 'munich' or 'all_data'
DATA_DIR=${3:-"embeddings_dir"}   # Default to 'embeddings_dir', can be 'embeddings_dir', 'data_dir', 'embeddings_dir_not_normalized'
DATA_SPECIFIER=${4:-"patches"}    # Default to 'patches'
PROCESSES=${5:-8}                 # Default to 8

if [ "$DATA_SPECIFIER" == "all" ]; then
  data_specifiers=("patches" "top_1_patches" "top_5_patches" "top_10_patches")
else
  data_specifiers=("$DATA_SPECIFIER")
fi

learning_rates=(0.001 0.01)
label_smoothing=(0.0 0.2 0.3)
warmup_epochs=(5 10)

for lr in "${learning_rates[@]}"; do
  for ls in "${label_smoothing[@]}"; do
    for wu in "${warmup_epochs[@]}"; do
      for specifier in "${data_specifiers[@]}"; do
        docker run --shm-size=400gb --gpus all \
          -it -u `id -u $USER` --rm \
          -v /home/ftoelkes/code:/mnt \
          -v /home/ftoelkes/preprocessed_data/train_data:/data \
          ftoelkes_lymphoma torchrun --standalone --nproc_per_node=$PROCESSES -m lymphoma.resnet_ddp.main \
          --total_epochs=50 --save_every=1 --validate_every=1 --batch_size=256 --offset=0 \
          --annotations_dir=inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0 \
          --patch_size="$PATCH_SIZE" --dataset="$DATASET" --data_specifier="$specifier" --data_dir="$DATA_DIR" \
          -vm="slide" --patience=10  \
          -lr=$lr -ls=$ls -wu=$wu -wl
      done
    done
  done
done


# example for single command without this script
# screen -dmS train_resnet sh -c 'docker run --shm-size=400gb --gpus all --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code:/mnt -v /home/ftoelkes/preprocessed_data/train_data:/data ftoelkes_lymphoma torchrun --standalone --nproc_per_node=1 -m lymphoma.resnet_ddp.main --total_epochs=1 --save_every=1 --validate_every=1 --batch_size=256 --offset=0 --annotations_dir=inner_5_fold_cv_patchesPercent_1.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0 --patch_size=1024um --data_dir=embeddings_dir --dataset=kiel --data_specifier=patches -vm=slide --patience=5 -wl -lr=0.001 -ls=0.1 ; exec bash'

