# change the screen command depending on which data set you want to sample (see below for single examples of command without this script) 
# Usage: /home/ftoelkes/code/lymphoma/data_preparation/run_normalize_slide_patches.sh [num_jobs] [slides_per_job] [start_slide] [mode]
# Example: /home/ftoelkes/code/lymphoma/data_preparation/run_normalize_slide_patches.sh 4 33 0 train


# Input parameters with fallback to defaults
NUM_JOBS=${1:-4}                  # Default to 4 jobs
SLIDES_PER_JOB=${2:-40}           # Default to 20 slides per job
START_SLIDE=${3:-0}               # Default to start from slide 0
MODE=${4:-"train"}                # Default to 'train'

# Validate mode to avoid invalid directory errors
if [ "$MODE" != "train" ] && [ "$MODE" != "test" ]; then
  echo "Error: Unsupported mode '$MODE'. Supported modes are 'train' and 'test'."
  exit 1
fi

# Main loop
for ((i=0; i<NUM_JOBS; i++))
do
  SCREEN_NAME="normalize_patches_${MODE}_$i"
  DOCKER_NAME="ftoelkes_run_${MODE}_$i"
  JOB_START_SLIDE=$((START_SLIDE + i * SLIDES_PER_JOB))
  
  screen -dmS "$SCREEN_NAME" sh -c \
    "docker run --shm-size=600gb --name $DOCKER_NAME -it -u \`id -u $USER\` --rm \
    -v /home/ftoelkes/code/lymphoma/data_preparation:/mnt \
    -v /home/ftoelkes/preprocessed_data/${MODE}_data:/data \
    ftoelkes_lymphoma python3 -m normalize_slide_patches --patch_size='1024um_munich_ltds' \
    --starting_slide=$JOB_START_SLIDE --nr_of_slides=$SLIDES_PER_JOB; exec bash"
done


