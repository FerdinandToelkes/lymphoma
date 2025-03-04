import os
import shutil
import argparse
import pandas as pd

from lymphoma.diagnosis_maps import LABELS_MAP_STRING_TO_INT

# Summary:
# This script is used to prepare the data directory for the single patch experiments.
# It takes the data directory in which the single patch files are stored in their corresponding slide directories and moves each slide directory to the corresponding class directory e.g one certain patch is found in data_dir/HL/23576-2025-0-HE-HL/patch_201382_coords.pt


# Usage on server:

"""
docker run -it --gpus \"device=0\" -it -u `id -u $USER` --rm -v /home/ftoelkes/preprocessed_data/test_data/1024um/kiel/patches:/data --rm -v /home/ftoelkes/code:/mnt ftoelkes_lymphoma python3 -m lymphoma.data_preparation.after_patchcraft.prepare_data_directory --new_path_to_data=/data/data_dir
"""

# or locally
"""python3 prepare_data_directory.py --path_to_data=/data --new_path_to_data=/data/data_dir"""

def add_arguments(parser: argparse.ArgumentParser):
    """Parse command line arguments.
    
    Args:
        parser (argparse.ArgumentParser): Parser object.

    Returns:
        argparse.ArgumentParser: Parser object with added arguments.
    """
    parser.add_argument("--path_to_data", type=str, default="/data", help="Path to data directory in which the single filed patches are stored in their corresponding slide directories. Default is /data.")
    parser.add_argument("--new_path_to_data", type=str, default="/data/data_dir", help="Path to data directory in which class directories containing the different slides with the single patches should be saved. Default is /data/data_dir.")
    return parser

def update_csv_files_with_stain_and_diagnosis(path_to_data: str):
    """Update the csv files in the given directory with the stain and diagnosis of the corresponding slides. 
    The data is structured as follows (directly after patchcraft):
    07312-2007-0-HE-HL.sqlite
        patch_0_..
        metadata.csv 
    07312-2007-1-HE-HL.sqlite
        patch_0_..
        metadata.csv

    Args:
        path_to_data (str): Path to the data directory containing the slide directories.
    """
    slides = os.listdir(path_to_data)
    slides = [slide for slide in slides if os.path.isdir(os.path.join(path_to_data, slide))]
    for slide in slides:
        # get stain and diagnosis from the slide name
        stain = slide.split("-")[-2]
        if stain != "HE":
            raise ValueError(f"Stain is not HE but {stain}.")
        diagnosis = slide.split("-")[-1]
        diagnosis = diagnosis.replace(".sqlite", "")

        csv_path = os.path.join(path_to_data, slide, "metadata.csv")
        if not os.path.exists(csv_path):
            raise ValueError(f"File {csv_path} does not exist.")
        df = pd.read_csv(csv_path)
        # check if stain column is filled with NaN
        if "stain" in df.columns and df["stain"].isnull().all():
            df["stain"] = stain
            df["diagnosis"] = LABELS_MAP_STRING_TO_INT[diagnosis]
            df.to_csv(csv_path, index=False)
        else:
            print(f"Column 'stain' is not empty in file {csv_path}, continuing without updating it.")


def make_class_directories(path_to_data: str):
    """ Create directories for the different classes in the given directory.

    Args:
        path_to_data (str): Path to the data directory in which the class directories
        should be created.
    """
    os.mkdir(os.path.join(path_to_data, "HL"))
    os.mkdir(os.path.join(path_to_data, "DLBCL"))
    os.mkdir(os.path.join(path_to_data, "CLL"))
    os.mkdir(os.path.join(path_to_data, "FL"))
    os.mkdir(os.path.join(path_to_data, "MCL"))
    os.mkdir(os.path.join(path_to_data, "LTDS"))

def move_slides_to_class_directories(path_to_data: str, slides: list) -> tuple:
    """
    Moves each slide directory to the corresponding class directory e.g one certain patch is found in
    data_dir/HL/00732-2009-0-HE-HL/patch_201382_coords.pt.

    Args:
        path_to_data (str): Path to the data directory in which the slide directories are stored.
        slides (list): List of slide directories.

    Returns:
        patch_counts (dict): Dictionary containing the number of patches per class.
        removed_slides (list): List of slide directories that were removed because they contained no patches.
    """
    removed_slides = []
    patch_counts = {"HL": 0, "DLBCL": 0, "CLL": 0, "FL": 0, "MCL": 0, "LTDS": 0}
    for i,slide in enumerate(slides):
        print(f"Handling slide directory {i}/{len(slides)}", end="\r")
        path_to_slide_dir = os.path.join(path_to_data, slide)
        files = os.listdir(path_to_slide_dir)
        # check if there are no pt files in the directory, if so delete the directory
        files = [file for file in files if file.endswith(".pt")]
        if len(files) == 0:
            print(f"No files found in slide directory {slide}. Deleting directory ...")
            shutil.rmtree(path_to_slide_dir)
            removed_slides.append(slide)
            continue
        # move slide directory to corresponding class directory. Slide directory name is something like 23576-2009-0-HE-CLL
        diagnosis = slide.split("-")[-1].upper()
        diagnosis = "LTDS" if diagnosis == "LTS" or diagnosis == "LYM" else diagnosis
        move_one_slide_to_class_directory(path_to_data, path_to_slide_dir, diagnosis)
        # count number of patches per class
        patch_counts[diagnosis] += len(files)
    return patch_counts, removed_slides

def move_one_slide_to_class_directory(path_to_data: str, path_to_slide_dir: str, diagnosis: str):
    """ Move one slide directory to the corresponding class directory.

    Args:
        path_to_data (str): Path to the data directory in which the class directories
        should be created.
        path_to_slide_dir (str): Path to the slide directory that should be moved.
        diagnosis (str): Diagnosis of the slide directory.
    """
    if diagnosis == "HL":
        shutil.move(path_to_slide_dir, os.path.join(path_to_data, "HL"))
    elif diagnosis == "DLBCL":
        shutil.move(path_to_slide_dir, os.path.join(path_to_data, "DLBCL"))
    elif diagnosis == "CLL":
        shutil.move(path_to_slide_dir, os.path.join(path_to_data, "CLL"))
    elif diagnosis == "FL":
        shutil.move(path_to_slide_dir, os.path.join(path_to_data, "FL"))
    elif diagnosis == "MCL":
        shutil.move(path_to_slide_dir, os.path.join(path_to_data, "MCL"))
    elif diagnosis == "LTDS" or diagnosis == "LTS" or diagnosis == "LYM":
        shutil.move(path_to_slide_dir, os.path.join(path_to_data, "LTDS"))
    else:
        raise Exception(f"Unknown diagnosis: {diagnosis}")
    
def create_info_file(path_to_data: str, patch_counts: dict, removed_slides: list):
    """ Create a INFO.md file with information about the data.

    Args:
        path_to_data (str): Path to the data directory in which the INFO.md file should be saved.
        patch_counts (dict): Dictionary containing the number of patches per class.
        removed_slides (list): List of slide directories that were removed because they contained no patches.
    """
    with open(os.path.join(path_to_data, "INFO.md"), "w") as f:
        f.write("# INFO\n")
        f.write(f"Total number of patches: {sum(patch_counts.values())}\n")
        f.write("\n")
        f.write("Number of patches per class:\n")
        for key, value in patch_counts.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("Removed slides:\n")
        for slide in removed_slides:
            f.write(f"{slide}\n")
        f.write("\n")
        f.write("NOTE:\nTo add new data, use the saved config.yaml file with a different seed (be carefull with seed 0):\n")

def move_class_directories_to_new_path(path_to_data: str, new_path_to_data: str):
    """ Move the class directories to the new path_to_data directory.

    Args:
        path_to_data (str): Path to the data directory in which the class directories
        are currently located.
        new_path_to_data (str): Path to the new data directory in which the class directories
        should be saved.
    """
    shutil.move(os.path.join(path_to_data, "HL"), new_path_to_data)
    shutil.move(os.path.join(path_to_data, "DLBCL"), new_path_to_data)
    shutil.move(os.path.join(path_to_data, "CLL"), new_path_to_data)
    shutil.move(os.path.join(path_to_data, "FL"), new_path_to_data)
    shutil.move(os.path.join(path_to_data, "MCL"), new_path_to_data)
    shutil.move(os.path.join(path_to_data, "LTDS"), new_path_to_data)


def main(path_to_data: str, new_path_to_data: str):
    """ Moves each slide directory to the corresponding class directory.
    
    Args:
        path_to_data (str): Path to the data directory in which the slide directories are stored.
        new_path_to_data (str): Path to the new data directory in which the class directories
        should be saved.
    """
    # update csv files with stain and diagnosis if necessary
    update_csv_files_with_stain_and_diagnosis(path_to_data)
    print("Updated csv files with stain and diagnosis if it was necessary.")
    # get all slide directories and filter out everything that is not a directory
    slides = os.listdir(path_to_data)
    slides = [slide for slide in slides if os.path.isdir(os.path.join(path_to_data, slide))]

    # check if there are directories for the different classes, if so raise error
    if "HL" in slides or "DLBCL" in slides or "CLL" in slides or "FL" in slides or "MCL" in slides or "LTDS" in slides:
        raise Exception("There are already directories for the different classes. You may delete them but be carefull with handling new data!")
    else:
        print("There are no directories for the different classes. Creating them ...")
        make_class_directories(path_to_data)
    
    # move each slide directory to corresponding class directory and count number of patches per class
    patch_counts, removed_slides = move_slides_to_class_directories(path_to_data, slides)

    # print number of patches per class and total number of patches
    print("\nNumber of patches per class:\n", patch_counts)
    print(f"Total number of patches: {sum(patch_counts.values())}")

    # create INFO.md file with information about the data
    create_info_file(path_to_data, patch_counts, removed_slides)

    # create and move the class directories to the new path_to_data directory
    if not os.path.exists(new_path_to_data):
        os.mkdir(new_path_to_data)
    else:
        raise Exception(f"Directory {new_path_to_data} already exists. Please delete it or choose a different path.")
    move_class_directories_to_new_path(path_to_data, new_path_to_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args.path_to_data, args.new_path_to_data)

