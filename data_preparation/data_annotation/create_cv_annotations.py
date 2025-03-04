import pandas as pd
import numpy as np
import argparse
import os

from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from data_annotation.create_slide_split import check_data_leakage_between_train_and_test_slides

"""
screen -dmS create_cv_annotations sh -c 'docker run -it --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/preprocessed_data/train_data:/data --rm -v /home/ftoelkes/code/lymphoma/data_preparation:/mnt ftoelkes_lymphoma python3 -m data_annotation.create_cv_annotations --percentage_of_patches=1.0 --shuffle_patches --probability_threshold=0 --patch_size="1024um" --dataset="kiel" ; exec bash' 
"""

def parse_arguments() -> dict:
    """ Parse command line arguments. 
    
    Returns:
        dict: Dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Create annotation files for training and evaluating on patch level level for cross validation.')
    parser.add_argument("--patch_size", type=str, default="1024um", help="Size of the patches. Default is 1024um.")
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply", "munich" or "all_data". (default: kiel)')
    parser.add_argument("--nr_of_folds", type=int, default=5, help="Number of folds for the (inner) cross validation. Default is 5.")
    parser.add_argument("--probability_threshold", type=int, default=0, help="Minimum tumor probability for the patches to be included (from 0 to 100). Default is 0.")
    parser.add_argument("--percentage_of_patches", type=float, default=100.0, help="Percentage of patches to be used for training, validation and testing in total. Default is 75.")
    parser.add_argument("--shuffle_patches", default=False, action="store_true", help="Whether to shuffle the patches of one slide. If True, the seed is used for reproducibility. Default is False.")
    parser.add_argument("--shuffle_dataframes", default=False, action="store_true", help="Whether to shuffle the rows of the dataframes. If True, the seed is used for reproducibility. Default is False.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for shuffling the dataframes.")
    parser.add_argument("--slides_split_name", type=str, default="train_test_slides_split", help="Name (prefix) of the csv file where the split of the slides is saved. Default is slides_split.")
    args = parser.parse_args()
    return dict(vars(args))




###############################################################################################################################
############################################## Splitting the data on patch level ##############################################
###############################################################################################################################

def split_data_on_patch_level(patch_size: str, slides_train: pd.DataFrame, train_names: list, indices: list, 
                              percentage_of_patches: float, shuffle_patches: bool, shuffle_dataframes: bool, 
                              probability_threshold: int, seed: int) -> pd.DataFrame:
    """Function to split the data on patch level.

    Args:
        slides_train: dataframe containing the slides in the train set
        train_names: list of the names of the slides in the train set
        indices: indices of the slides in the train set
        percentage_of_patches: percentage of patches to be used for training and validation in total
        shuffle_patches: whether to shuffle the patches of one slide
        shuffle_dataframes: whether to shuffle the rows of the dataframes
        probability_threshold: minimum tumor probability for the patches to be included
        seed: seed for shuffling the dataframes

    Returns:
        df: dataframe filled with the patches of the target slides
    """
    # get the corresponding dataframes from the slides_train dataframe
    desired_slides = slides_train[slides_train["filename"].isin([train_names[i] for i in indices])]
    # reindex the dataframes
    desired_slides = desired_slides.reset_index(drop=True)
    if len(desired_slides) == 0:
        raise ValueError("No slides found for the specified indices. Please check the indices and rerun the script.")
    # create empty dataframes
    df = pd.DataFrame(columns=["dataset", "filename", "label", "class"])
    # fill the dataframes with the patches belonging to the slides specfied as train, test and validation correspondingly 
    df = fill_dataframe_with_corresponding_patches(patch_size, desired_slides, df, percentage_of_patches, 
                                                            shuffle_patches, probability_threshold, seed)
    # randomly reorder the rows of the dataframes -> normally done during training
    if shuffle_dataframes:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

def fill_dataframe_with_corresponding_patches(patch_size: str, target_slides: pd.DataFrame, target_dataframe: pd.DataFrame, 
                                              percentage_of_patches: float, shuffle_patches: bool, 
                                              probability_threshold: int, seed: int) -> pd.DataFrame:
    """Function to fill the target dataframe with the corresponding patches of the target slides.
    
    Args:
        target_slides: dataframe containing the target slides
        target_dataframe: dataframe to be filled with the patches of the target slides
        path_to_data: path to the data directory
        percentage_of_patches: percentage of patches to be used for training, validation and testing in total
        shuffle_patches: whether to shuffle the patches
        probability_threshold: minimum tumor probability for the patches to be included
        seed: seed for shuffling the patches

    Returns:
        target_dataframe: dataframe filled with the patches of the target slides
    """
    # iterate over the target slides -> iterrows() returns an iterator yielding index and row data as a Series
    for _, row in tqdm(target_slides.iterrows(), total=len(target_slides)):
        # get all desired patches for the current slide (exclude the ones that are not .pt files)
        slide_path = os.path.join("/data", patch_size, row["dataset"], "patches", "data_dir", row["filename"])

        if not os.path.exists(slide_path): # since slides do not have to be available for any patch size
            print(f"Path {slide_path} does not exist. Continuing with the next slide.")
            continue
        
        # get patches, filter out patches below the probability threshold and sort them
        patches = get_patches_and_filter_out_patches_below_threshold(slide_path, probability_threshold, shuffle_patches, seed)
        
        # determine the number of patches to be used
        nr_of_patches = int(len(patches) * percentage_of_patches)
        print(f"Using {nr_of_patches} patches for slide {row['filename']} of dataset {row['dataset']}.")
        patches = patches[:nr_of_patches]
        patches = [os.path.join(row["filename"], patch) for patch in patches]
       
        # create a dataframe for the patches of the current slide and set the label and class
        df_patches = pd.DataFrame(patches, columns=["filename"])
        df_patches["dataset"] = row["dataset"]
        df_patches["label"] = int(row["label"]) 
        df_patches["class"] = row["class"]

        # add the patches to the target dataframe
        target_dataframe = pd.concat([target_dataframe, df_patches])
    return target_dataframe

def get_patches_and_filter_out_patches_below_threshold(slide_path: str, probability_threshold: int, shuffle: bool, seed: int) -> list:
    """Function to get the patches of a slide and filter out the patches below the probability threshold and shuffle them if desired.
    
    Args:
        slide_path: path to the slide
        probability_threshold: minimum tumor probability for the patches to be included
        shuffle: whether to shuffle the patches
        seed: seed for shuffling the patches

    Returns:
        patches: list of the patches of the slide
    """
    patches = os.listdir(slide_path)
    patches = [patch for patch in patches if patch.startswith("patch") and patch.endswith(".pt")]
    
    # filter out patches with tumor probability less than desired threshold 
    # note: the patches are named like "patch_x_y_..._tp=75.pt"
    if probability_threshold > 100 or probability_threshold < 0:
        raise ValueError("The minimum tumor probability must be between 0 and 100.")
    elif probability_threshold > 0:
        patches = [patch for patch in patches if int(patch.split("tp=")[-1].split(".")[0]) >= probability_threshold]
    
    patches.sort() # for reproducibility
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(patches)
    return patches

def perform_shuffle_of_dataframes(df_train, df_val, seed) -> tuple:
    """Function to shuffle the dataframes for training, validation and testing.
    
    Args:
        df_train: dataframe containing the training data
        df_val: dataframe containing the validation data
        seed: seed for shuffling the dataframes

    Returns:
        df_train: shuffled dataframe containing the training data
        df_val: shuffled dataframe containing the validation data
    """
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True) # 
    df_val = df_val.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_train, df_val

def save_load_and_print_dataframes(df_train: pd.DataFrame, df_val: pd.DataFrame, path_to_output: str, k: int):
    """Function to save the dataframes to csv files, reload them and print them.
    
    Args:
        df_train: dataframe containing the training data
        df_val: dataframe containing the validation data
        path_to_output: path to the output directory
        k: index of the fold
    """
    df_train.to_csv(os.path.join(path_to_output, f"train_{k}.csv"), index=False)
    df_val.to_csv(os.path.join(path_to_output, f"val_{k}.csv"), index=False)
    # reload them and print them to check if they are saved correctly
    df_train_reload = pd.read_csv(os.path.join(path_to_output, f"train_{k}.csv"))
    df_val_reload = pd.read_csv(os.path.join(path_to_output, f"val_{k}.csv"))
    print(f"df_train", df_train_reload)
    print(f"df_val", df_val_reload)
    

####################################################################################################################################
###################################################### Helper functions for main ###################################################
####################################################################################################################################


def create_output_dir(output_dir: str, nr_of_folds: int, probabilty_threshold: int, percentage_of_patches: float, 
                      shuffle_patches: bool, shuffle_dataframes: bool, seed: int) -> tuple:
    """Function to create the output directory for the annotation files.
    
    Args:
        output_dir: path to the output directory
        nr_of_folds: number of folds for the cross validation
        probabilty_threshold: minimum tumor probability for the patches to be included
        percentage_of_patches: percentage of patches to be used for training and validation in total
        shuffle_patches: whether to shuffle the patches of one slide
        shuffle_dataframes: whether to shuffle the rows of the dataframes
        seed: seed for shuffling the dataframes

    Returns:
        path_to_output: path to the output directory
        dir_exists: whether the output directory already exists
    """
    dir_exists = False
    # create description for the output directory
    description = f"inner_{nr_of_folds}_fold_cv_patchesPercent_{percentage_of_patches*100}_shufflePatches_{shuffle_patches}_shuffleDataframes_{shuffle_dataframes}_tumorThreshold_{probabilty_threshold}_seed_{seed}"
    os.makedirs(output_dir, exist_ok=True)
    path_to_output = os.path.join(output_dir, description)
    if os.path.exists(path_to_output):
        print(f"Output directory {path_to_output} already exists. Loading annotations from this directory.")
        dir_exists = True
    else:
        print(f"Creating output directory {path_to_output}.")
        os.mkdir(path_to_output)
    return path_to_output, dir_exists

def prepare_cross_validation_split(slides_split_path: str) -> tuple:
    """Function to prepare the cross validation split of the slides.
    
    Args:
        slides_split_path: path to the csv file containing the split of the slides into train and test set.
    
    Returns:
        slides_train: dataframe containing the slides in the train set
        train_names: list of the names of the slides in the train set
        train_groups: list of the patient ids of the slides in the train set
        train_labels: list of the labels of the slides in the train set
    """
    # get slides where train is set to 1
    df_slides = pd.read_csv(slides_split_path)
    slides_train = df_slides[df_slides["train"] == 1]
    print(f"Number of slides in train set: {len(slides_train)}")

    # get train slide names and the corresponding labels in seperate lists
    train_names = slides_train["filename"].tolist()
    train_groups = [name.split("-")[0] for name in train_names] # patient ids
    train_labels = slides_train["label"].tolist()
    return slides_train, train_names, train_groups, train_labels

####################################################################################################################################
################################################# Check for data leakage ###########################################################
####################################################################################################################################

def check_splits_for_data_leakage(patch_size: str, annotations_dir: str, nr_of_folds: int, path_to_slides_split: str) -> None:
    """ Check if the splits used for testing are without any data leakage.

    Args:
        patch_size (str): Size of the patches
        annotations_dir (str): Path to the directory where the different annotations files are located which is needed for the dataloader
        nr_of_folds (int): Number of cross-validation folds
        path_to_slides_split (str): Path to the csv file with the train and test slides
    """
    print("Checking for data leakage between train and test slides and between train/val and test slides in cross-validation folds")
    train_slide_names, test_slide_names = check_data_leakage_between_train_and_test_slides(path_to_slides_split)
    cv_base_path = os.path.join("/train_data",f"{patch_size}_annotations", annotations_dir) 
    check_data_leakage_in_cv_folds(cv_base_path, nr_of_folds, train_slide_names, test_slide_names)
    print("No data leakage found between train and test slides and between train/val and test slides in cross-validation folds")

def check_data_leakage_in_cv_folds(cv_base_path: str, nr_of_folds: int, train_slide_names: set, test_slide_names: set) -> None:
    """ Check if the train and val slides in the cross-validation folds are disjoint and if they only contain slides from the train slides 
        and thus should be disjoint from the test slides.

    Args:
        cv_base_path (str): Base path to the annotations directory containing the cross-validation splits
        nr_of_folds (int): Number of cross-validation folds
        train_slide_names (set): Names of the train slides
        test_slide_names (set): Names of the test slides
    """
    for k in range(nr_of_folds):
        df_cv_train = pd.read_csv(os.path.join(cv_base_path, f"train_{k}.csv"))
        df_cv_val = pd.read_csv(os.path.join(cv_base_path, f"val_{k}.csv"))
        cv_train_slide_names = set([os.path.join(name.split("/")[0], name.split("/")[1]) for name in df_cv_train["filename"]])
        cv_val_slide_names = set([os.path.join(name.split("/")[0], name.split("/")[1]) for name in df_cv_val["filename"]])
        print(f"Number of unique train slides in fold {k}: {len(cv_train_slide_names)}")
        print(f"Number of unique val slides in fold {k}:   {len(cv_val_slide_names)}")

        cv_train_and_val_slide_names = cv_train_slide_names.union(cv_val_slide_names)
        # note that we are only interested if slides from the train data are not in the cv train and val data
        diff_train_and_cv_train = cv_train_and_val_slide_names.difference(train_slide_names) # other direction is not interesting

        if len(cv_val_slide_names.intersection(cv_train_slide_names)) > 0:
            raise ValueError(f"Train and val slides are not disjoint in fold {k}, i.e., there is data leakage in this fold")
        # the follwowing two checks should be equivalent since we already checked that train and test slides are disjoint
        if len(diff_train_and_cv_train) > 0:
            raise ValueError(f"There are train or val slides in fold {k} that are not in the train slides on which the cross-validation is based on")
        if len(cv_train_and_val_slide_names.intersection(test_slide_names)) > 0:
            raise ValueError(f"Train and val slides in fold {k} are not disjoint from the test slides, i.e., there is data leakage between train/val from the cross-validation and the final test slides")

####################################################################################################################################
############################################################## Main function #######################################################
####################################################################################################################################

def main(patch_size: str, dataset: str, nr_of_folds: int, probability_threshold: int, percentage_of_patches: float, 
         shuffle_patches: bool, shuffle_dataframes: bool, seed: int, slides_split_name: str):
    """Main function to create the annotation files for training and validation on patch level.
    
    Args:
        patch_size (str): Size of the patches
        dataset (str): Name of the dataset
        nr_of_folds (int): Number of folds for the (inner) cross validation
        probability_threshold (int): Minimum tumor probability for the patches to be included
        percentage_of_patches (float): Percentage of patches to be used for training, validation and testing in total
        shuffle_patches (bool): Whether to shuffle the patches of one slide
        shuffle_dataframes (bool): Whether to shuffle the rows of the dataframes
        seed (int): Seed for shuffling the dataframes
        slides_split_name (str): Name of the csv file where the split of the slides is saved
    """
    # set necessary paths
    output_dir = f"/data/{patch_size}/annotations/{dataset}"
    slides_split_path = os.path.join(output_dir, slides_split_name + ".csv")
    if not os.path.exists(slides_split_path):
        raise ValueError(f"Path {slides_split_path} does not exist. Please make sure to run the script create_slide_split.py first.")

    # create output directory recursively if it does not exist
    path_to_output, dir_exists = create_output_dir(output_dir, nr_of_folds, probability_threshold, percentage_of_patches, 
                                        shuffle_patches, shuffle_dataframes, seed)
    if not dir_exists:
        print(f"Output directory: {path_to_output}")

        # load the split of the slides
        slides_train, train_names, train_groups, train_labels = prepare_cross_validation_split(slides_split_path)
        
        sgkf = StratifiedGroupKFold(n_splits=5)
        for k, (train_indices, test_indices) in enumerate(sgkf.split(train_names, train_labels, groups=train_groups)):
            df_train = split_data_on_patch_level(patch_size, slides_train, train_names, train_indices, percentage_of_patches, 
                                                shuffle_patches, shuffle_dataframes, probability_threshold, seed)
            df_val = split_data_on_patch_level(patch_size, slides_train, train_names, test_indices, percentage_of_patches, 
                                                shuffle_patches, shuffle_dataframes, probability_threshold, seed)
            
            save_load_and_print_dataframes(df_train, df_val, path_to_output, k)

    # check for data leakage between train and test slides and between train/val and test slides in cross-validation folds
    check_splits_for_data_leakage(patch_size, path_to_output, nr_of_folds, slides_split_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(**args)