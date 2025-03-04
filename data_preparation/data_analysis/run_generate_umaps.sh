#!/bin/bash

# Example usage: screen -dmS generate_umaps sh -c '/home/ftoelkes/code/lymphoma/data_preparation/data_analysis/run_generate_umaps.sh 20 0.1 ; exec bash'


# Input arguments
N_NEIGHBOR=${1:-15}     # Default to 15 neighbors
MIN_DIST=${2:-0.1}      # Default to 0.1 minimum distance

# Define classes and datasets
CLASSES=("CLL" "DLBCL" "FL" "HL" "LTDS" "MCL")
DATASETS=("kiel" "munich")

# Combine given classes: CLL DLBCL FL LTDS MCL -> CLL,DLBCL,FL,LTDS,MCL
COMBINED_CLASSES=$(IFS=,; echo "${CLASSES[*]}")
# Combine given datasets: kiel munich -> kiel,munich
COMBINED_DATASETS=$(IFS=,; echo "${DATASETS[*]}")

# Loop for separate datasets
for DATASET in "${DATASETS[@]}"
do
    for CLASS in "${CLASSES[@]}"
    do
        if [ "$CLASS" == "HL" ] && [ "$DATASET" == "munich" ]; then
            echo "Skipping class $CLASS for dataset $DATASET"
            continue
        fi

        echo "Processing class $CLASS for dataset $DATASET"

        docker run --shm-size=400gb --gpus all --name "ftoelkes_run_${DATASET}_${CLASS}" -it -u "$(id -u $USER)" \
        --rm -v /home/ftoelkes/code/lymphoma/data_preparation/data_analysis:/mnt \
        -v /home/ftoelkes/preprocessed_data/test_data:/data \
        ftoelkes_lymphoma python3 -m generate_umaps --patch_size=1024um --data_dir=embeddings_dir  \
        --data_specifier=patches --unique_slides --datasets="$DATASET" --classes="$CLASS" --n_neighbors="$N_NEIGHBOR" --min_dist="$MIN_DIST"
    done

    echo "Processing combined classes $COMBINED_CLASSES for dataset $DATASET"

    docker run --shm-size=400gb --gpus all --name "ftoelkes_run_${DATASET}_combined" -it -u "$(id -u $USER)" \
    --rm -v /home/ftoelkes/code/lymphoma/data_preparation/data_analysis:/mnt \
    -v /home/ftoelkes/preprocessed_data/test_data:/data \
    ftoelkes_lymphoma python3 -m generate_umaps --patch_size=1024um --data_dir=embeddings_dir  \
    --data_specifier=patches --unique_slides \
    --datasets="$DATASET" --classes="$COMBINED_CLASSES" --n_neighbors="$N_NEIGHBOR" --min_dist="$MIN_DIST"
done

# Loop for combined datasets
for CLASS in "${CLASSES[@]}"
do
    if [ "$CLASS" == "HL" ]; then
        echo "Skipping class $CLASS for combined datasets $COMBINED_DATASETS"
        continue
    fi

    echo "Processing class $CLASS for combined datasets $COMBINED_DATASETS"

    docker run --shm-size=400gb --gpus all --name "ftoelkes_run_combined_${CLASS}" -it -u "$(id -u $USER)" \
    --rm -v /home/ftoelkes/code/lymphoma/data_preparation/data_analysis:/mnt \
    -v /home/ftoelkes/preprocessed_data/test_data:/data \
    ftoelkes_lymphoma python3 -m generate_umaps --patch_size=1024um --data_dir=embeddings_dir  \
    --data_specifier=patches --unique_slides \
    --datasets="$COMBINED_DATASETS" --classes="$CLASS" --n_neighbors="$N_NEIGHBOR" --min_dist="$MIN_DIST"
done
