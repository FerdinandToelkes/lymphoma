#!/bin/bash

# Example usage (from code directory): ./lymphoma/data_preparation/data_analysis/run_visualize_umaps.sh

CLASSES=("CLL" "DLBCL" "FL" "HL" "LTDS" "MCL")
DATASETS=("kiel" "munich")
# combine given classes: CLL DLBCL FL LTDS MCL -> CLL,DLBCL,FL,LTDS,MCL
COMBINED_CLASSES=$(IFS=,; echo "${CLASSES[*]}")
# combine given datasets: kiel munich -> kiel,munich
COMBINED_DATASETS=$(IFS=,; echo "${DATASETS[*]}")

# loop for seperate datasets
for DATASET in "${DATASETS[@]}"
do
    for CLASS in "${CLASSES[@]}"
    do
    if [ "$CLASS" == "HL" ] && [ "$DATASET" == "munich" ]; then
        echo "Skipping class $CLASS for dataset $DATASET"
        continue
    fi
    echo "Processing class $CLASS for dataset $DATASET"
    python3 -m lymphoma.data_preparation.data_analysis.visualize_umaps \
    --unique_slides --datasets=$DATASET --classes=$CLASS
    done
echo "Processing combined classes $COMBINED_CLASS for dataset $DATASET"
python3 -m lymphoma.data_preparation.data_analysis.visualize_umaps \
--unique_slides --datasets=$DATASET --classes=$COMBINED_CLASSES
done

# loop for combined datasets
for CLASS in "${CLASSES[@]}"
do
    if [ "$CLASS" == "HL" ]; then
        echo "Skipping class $CLASS for combined datasets $COMBINED_DATASETS"
        continue
    fi
    echo "Processing class $CLASS"
    python3 -m lymphoma.data_preparation.data_analysis.visualize_umaps \
    --unique_slides --datasets=$COMBINED_DATASETS --classes=$CLASS
done
