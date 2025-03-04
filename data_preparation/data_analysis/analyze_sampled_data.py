import pandas as pd
import numpy as np
import argparse
import os

from lymphoma.data_preparation.data_annotation.create_slide_split import get_target_data_dirs

# this script has similar purpose as analyze_data_distributions.py, but originated after switching to using multiple data sets, i.e. its newer

# To run this script on the server:
"""
docker run --shm-size=100gb --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code:/mnt  -v /home/ftoelkes/preprocessed_data/train_data:/data  ftoelkes_lymphoma python3 -m lymphoma.data_preparation.data_analysis.analyze_sampled_data --dataset kiel
"""

def parse_arguments() -> dict:
    """ Parse command line arguments. 
    
    Returns:
        dict: Dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Create annotation files for training and testing on slide level.')
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply", "munich" or "all_data". (default: kiel)')
    args = parser.parse_args()
    return dict(vars(args))

def analyze_sampled_data(target_data_dirs: list) -> dict:
    """ Analyze the sampled data, by counting the number of slides, unique slides, patches and unique patches per class and data directory.

    Args:
        target_data_dirs (list): list of data directories that will be used for training later

    Returns:
        results (dict): dictionary containing the number of slides, unique slides, patches and unique patches per class and data directory
    """
    # setup
    slides_per_class_per_data_dir = {d: {} for d in target_data_dirs}
    unique_slides_per_class_per_data_dir = {d: {} for d in target_data_dirs}
    patches_per_class_per_data_dir = {d: {} for d in target_data_dirs}
    unique_patches_per_class_per_data_dir = {d: {} for d in target_data_dirs}
    classes = os.listdir(os.path.join("/data", target_data_dirs[0], "data_dir"))

    for target_dir in target_data_dirs:
        for c in classes:
            path_to_class_dir = os.path.join("/data", target_dir, "data_dir", c)
            slides = os.listdir(path_to_class_dir)
            slides = [slide for slide in slides if os.path.isdir(os.path.join(path_to_class_dir, slide))]
            slides_per_class_per_data_dir[target_dir][c] = len(slides)
            # filter such that only one slide per patient is counted
            unique_slides = [s for s in slides if s.split("-")[2] == "0"]
            unique_slides_per_class_per_data_dir[target_dir][c] = len(unique_slides)

            patches_per_class_per_data_dir = count_available_patches(slides, path_to_class_dir, target_dir, c, patches_per_class_per_data_dir)
            unique_patches_per_class_per_data_dir = count_available_patches(unique_slides, path_to_class_dir, target_dir, c, unique_patches_per_class_per_data_dir)
        
        slides_per_class_per_data_dir[target_dir]["total"] = np.sum(list(slides_per_class_per_data_dir[target_dir].values()))
        unique_slides_per_class_per_data_dir[target_dir]["total"] = np.sum(list(unique_slides_per_class_per_data_dir[target_dir].values()))
        patches_per_class_per_data_dir[target_dir]["total"] = np.sum(list(patches_per_class_per_data_dir[target_dir].values()))
        unique_patches_per_class_per_data_dir[target_dir]["total"] = np.sum(list(unique_patches_per_class_per_data_dir[target_dir].values()))

    return {
        "slides_per_class_per_data_dir": slides_per_class_per_data_dir,
        "unique_slides_per_class_per_data_dir": unique_slides_per_class_per_data_dir,
        "patches_per_class_per_data_dir": patches_per_class_per_data_dir,
        "unique_patches_per_class_per_data_dir": unique_patches_per_class_per_data_dir
    }

def count_available_patches(slides: list, path_to_class_dir: str, target_dir: str, current_class: str, nr_of_data_dict: dict) -> dict:
    """ Count the number of patches in a class. 
    
    Args:
        slides (list): list of slides in the class
        path_to_class_dir (str): path to the class directory
        target_dir (str): name of the data directory
        current_class (str): name of the class
        nr_of_data_dict (dict): dictionary containing the number of patches per class

    Returns:
        nr_of_data_dict (dict): dictionary containing the updated number of patches per class
    """
    nr_of_patches_of_class = 0
    for slide in slides:
        patches = os.listdir(os.path.join(path_to_class_dir, slide))
        patches = [patch for patch in patches if patch.startswith("patch") and patch.endswith(".pt")]
        nr_of_patches_of_class += len(patches)

    nr_of_data_dict[target_dir][current_class] = nr_of_patches_of_class
    return nr_of_data_dict

def print_and_save_and_results_as_csv(data: dict, path: str, filename: str) -> None:
    """ Print and save results as csv file in results directory. 
    
    Args:
        data (dict): dictionary containing the data to be saved
        path (str): path to the directory where the file should be saved
        filename (str): name of the file
    """
    df = pd.DataFrame(data)
    print(f"df for {filename}:\n{df}\n")
    df.to_csv(f"{path}/{filename}.csv")




def main(patch_size: str, dataset: str):
    """ Main function to analyze the sampled data.

    Args:
        patch_size (str): size of the patches
        dataset (str): name of the dataset
    """
    target_data_dirs = get_target_data_dirs(patch_size, dataset)
    
    # analyze data
    results = analyze_sampled_data(target_data_dirs)

    
    # save results to file and print them
    path = f"/mnt/lymphoma/data_preparation/data_analysis/results/{patch_size}/{dataset}/distributions"
    os.makedirs(path, exist_ok=True)
    print_and_save_and_results_as_csv(results["unique_slides_per_class_per_data_dir"], path, "unique_slides")
    print_and_save_and_results_as_csv(results["unique_patches_per_class_per_data_dir"], path, "unique_patches")
    print_and_save_and_results_as_csv(results["slides_per_class_per_data_dir"], path, "slides")
    print_and_save_and_results_as_csv(results["patches_per_class_per_data_dir"], path, "patches")
    




if __name__ == "__main__":
    args = parse_arguments()
    main(**args)
    