import os

import torch
import argparse
from tqdm import tqdm

# normalize train patches
"""
screen -dmS normalize_patches_train sh -c 'docker run --shm-size=600gb --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code/lymphoma/data_preparation/after_patchcraft:/mnt -v /home/ftoelkes/preprocessed_data/train_data:/data ftoelkes_lymphoma python3 -m normalize_slide_patches --patch_size="1024um" --dataset="kiel" --starting_slide=0 --nr_of_slides=20; exec bash'
"""

# normalize test patches
"""
screen -dmS normalize_patches_test sh -c 'docker run --shm-size=400gb --gpus \"device=1\" --name ftoelkes_run1 -it -u `id -u $USER` --rm -v /home/ftoelkes/code/lymphoma/data_preparation/after_patchcraft:/mnt -v /home/ftoelkes/preprocessed_data/test_data:/data ftoelkes_lymphoma python3 -m normalize_slide_patches --patch_size="1024um" --dataset="kiel" --starting_slide=0 --nr_of_slides=50; exec bash'
"""

def parse_arguments() -> dict:
    """ Parse command line arguments. 
    
    Returns:
        dict: Dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Prepare normalization of patches.')
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply" or "munich". (default: kiel)')
    parser.add_argument('--starting_slide', default=0, type=int, help='Index of the first slide to start with. (default: 0)')
    parser.add_argument('--nr_of_slides', default=1, type=int, help='Number of slides to process. (default: 1)')
    args = parser.parse_args()
    return vars(args)


def main(patch_size: str, dataset: str, starting_slide: int, nr_of_slides: int):
    """ Normalize the patches of the slides of the specified dataset. The mean and std of the patches of each slide are calculated and saved in a .pt file in the slide directory.

    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply" or "munich".
        starting_slide (int): Index of the first slide to start with.
        nr_of_slides (int): Number of slides to process.
    """
    # setup path to directory with different classes containing the slide directories with the patches
    path_to_class_dirs = os.path.join("/data", patch_size, dataset, "patches", "data_dir")
    print(f"path_to_class_dirs: {path_to_class_dirs}")
    path_to_slides = []
    # get all slide paths from the different classes
    for c in os.listdir(path_to_class_dirs):
        class_dir = os.path.join(path_to_class_dirs, c)
        slides_in_class = os.listdir(class_dir)
        path_to_slides.extend([os.path.join(class_dir, slide) for slide in slides_in_class]) 
    print(f"Found {len(path_to_slides)} slides in {path_to_class_dirs}")
    
    # process only a subset of the slides
    path_to_slides.sort()
    max_nr_of_slides = len(path_to_slides)
    stop_slide = starting_slide + nr_of_slides
    if starting_slide >= max_nr_of_slides:
        raise ValueError(f"starting_slide ({starting_slide}) is greater than the number of slides ({max_nr_of_slides})")
    if stop_slide > max_nr_of_slides:
        stop_slide = max_nr_of_slides
    path_to_slides = path_to_slides[starting_slide:stop_slide]

    for slide_path in tqdm(path_to_slides, total=len(path_to_slides)):
        path = os.path.join(slide_path, "mean_std.pt")
        if os.path.exists(path):
            print(f"Mean and Std already calculated for slide at {slide_path}")
            continue

        patches_of_slide = []
        patches = os.listdir(slide_path)
        patches = [patch for patch in patches if patch.endswith(".pt") and patch.startswith("patch")]
        len_patches = len(patches)
        
        for patch in tqdm(patches, total=len_patches, desc=f"Processing {slide_path}"):
            image, label = torch.load(os.path.join(slide_path, patch))
            # normalizing the image ensures more precise mean and std calculation and needed for pytorch mean and std calculation
            patches_of_slide.append(image)  # image.float().div(255)

        # stack the remaining patches
        stacked_patches = torch.stack(patches_of_slide)

        print(f"stacked_patches.shape: {stacked_patches.shape}") # [num_patches, channels, height, width] 
        # Calculate mean and std, dim=[0, 2, 3]: calculate over all patches, height and width for EACH channel
        mean = stacked_patches.float().mean(dim=[0, 2, 3]) # more memory efficient than mean = torch.mean(stacked_patches.float(), dim=[0, 2, 3])
        std = stacked_patches.float().std(dim=[0, 2, 3])
        mean = mean.div(255)
        std = std.div(255)
        print(f"Slide at {slide_path} before Normalization: Mean {mean}, Std {std}")
        print()


        torch.save((mean, std), path)
    

if __name__ == '__main__':
    args = parse_arguments()
    main(**args)
