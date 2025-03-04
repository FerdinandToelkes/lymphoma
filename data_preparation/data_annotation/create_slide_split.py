import pandas as pd
import numpy as np
import argparse
import os

from pamly import Diagnosis

"""
docker run -it --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/preprocessed_data/train_data:/data --rm -v /home/ftoelkes/code/lymphoma/data_preparation:/mnt ftoelkes_lymphoma python3 -m data_annotation.create_slide_split --shuffle_slides --patch_size 1024um --dataset all_data --test_slides_per_class 15
"""

def parse_arguments() -> dict:
    """ Parse command line arguments. 
    
    Returns:
        dict: Dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Create annotation files for training and testing on slide level.')
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply", "munich" or "all_data". (default: kiel)')
    parser.add_argument('-ts', '--test_slides_per_class', default=5, type=int, help='Number of slides for testing per class. Default is 5.')
    parser.add_argument("--shuffle_slides", default=False, action="store_true", help="Whether to shuffle the slides for the corresponding slides split csv file. If True, the seed is used for reproducibility. Default is False.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for shuffling the dataframes.")
    parser.add_argument("--slides_split_name", type=str, default="train_test_slides_split", help="Name (prefix) of the csv file where the split of the slides is saved. Default is slides_split.")
    args = parser.parse_args()
    return dict(vars(args))

#####################################################################################################################################
############################## Function for getting number of testing slides per class and data dir #################################
#####################################################################################################################################

def get_slides_per_class_per_data_dir(df: pd.DataFrame, test_slides_per_class: int, target_data_dirs: list) -> dict:
    """Function to compute how many unique slides per class and per dataset should be used for testing.

    Args:
        df (pd.DataFrame): Dataframe containing the total number of unique slides per class and per data directory
        test_slides_per_class (int): Desired number of slides for testing per class
        target_data_dirs (list): List of data directories that will be used for training later

    Returns:
        test_slides_per_class_and_data_dir (dict): Dictionary containing the number of slides per class and per data directory that should be used for testing
    """
    classes = df.index
    test_slides_per_class_and_data_dir = {d : {c : test_slides_per_class for c in classes} for d in target_data_dirs}
    for c in classes:
        slides_per_dir = {}
        # first get the number of slides per data directory
        for target_data_dir in target_data_dirs:
            nr_of_slides = df.loc[c, target_data_dir]
            slides_per_dir[target_data_dir] = nr_of_slides
        # use the number of slides per data directory to compute the number of test slides per data directory
        nr_of_missing_slides = test_slides_per_class
        for target_data_dir in target_data_dirs[1:]:
            nr_test_slides_of_data_dir = round(slides_per_dir[target_data_dir] / np.sum(list(slides_per_dir.values())) * test_slides_per_class)
            test_slides_per_class_and_data_dir[target_data_dir][c] = nr_test_slides_of_data_dir
            nr_of_missing_slides -= nr_test_slides_of_data_dir
        
        # take rest of slides from first data directory since it contains slides from all classes
        if nr_of_missing_slides < 0:
            raise ValueError("Number of test slides per class and per data directory must be at least 0.")
        test_slides_per_class_and_data_dir[target_data_dirs[0]][c] = nr_of_missing_slides
    return test_slides_per_class_and_data_dir
    

#####################################################################################################################################
############################################# Functions for splitting the slides ####################################################
#####################################################################################################################################

def create_train_balanced_test_slide_splits(test_slides_per_class_and_data_dir: dict, shuffle_slides: bool, seed: int, 
                                            slides_split_path: str):
    """
    Function to create the split of the slides into train and test set. The split is saved in a csv file and is later 
    also used to split the data on patch level. Essentially, the function takes a fixed number of unique slides for 
    testing for each class and the rest is split into the train set.

    Args:
        test_slides_per_class_and_data_dir (list): List of dictionaries containing the number of slides per class 
                                                and per data directory that should be used for testing
        shuffle_slides (bool): Whether to shuffle the slides
        seed (int): Seed for shuffling the slides
        slides_split_path (str): Path to the csv file where the split of the slides is saved
    """
    # setup
    target_data_dirs = list(test_slides_per_class_and_data_dir.keys())
    classes = list(test_slides_per_class_and_data_dir[target_data_dirs[0]].keys())
    df_slides = pd.DataFrame(columns=["dataset", "filename", "label", "class", "train", "test"])
    
    for target_data_dir in target_data_dirs:
        path_to_data = f"/data/{target_data_dir}/data_dir"
        for c in classes:
            path_to_class = os.path.join(path_to_data, c)
            # get filenames with relative path to the slide 
            slides = os.listdir(path_to_class)
            # ensure that only directories are considered as slides
            slides = [slide for slide in slides if os.path.isdir(os.path.join(path_to_class, slide))]
            if len(slides) == 0:
                print(f"No slides found in {path_to_class}, continuing with the next class.")
                continue
        
            # extract slide names that stem from the same patient -> 81599-2020-0-HE-CLL, 81599-2020-1-HE-CLL, 81599-2020-2-HE-CLL etc.
            ids_and_nr_of_duplicate_slides = get_ids_and_nr_of_duplicates(slides)

            # remove the slides with the same patient id from the list of slides and shuffle them if desired
            slides_one_per_patient = [slide for slide in slides if slide.split("-")[2] == "0"]
            slides_one_per_patient = [os.path.join(c, slide) for slide in slides_one_per_patient]
            if shuffle_slides:
                np.random.seed(seed)
                np.random.shuffle(slides_one_per_patient)

            # create a dataframe for the slides of one class with added prefix for the class
            slides = [os.path.join(c, slide) for slide in slides]
            dataset = target_data_dir.split("/")[1]
            df_slides_one_class = setup_dataframe_for_one_class(slides, c, dataset)

            # mark the slides with the corresponding split and add them to the dataframe
            mark_slides_with_train_test_split(
                slides_one_per_patient, 
                ids_and_nr_of_duplicate_slides, 
                df_slides_one_class, 
                test_slides_per_class_and_data_dir[target_data_dir][c]
            )

            # sort the dataframe by the one hot encoded columns
            df_slides_one_class = df_slides_one_class.sort_values(by=["train", "test"], ascending=False)
            df_slides = pd.concat([df_slides, df_slides_one_class])
    
    # save the split df_slides
    df_slides.to_csv(slides_split_path, index=False)
    print(f"Saved the split of the slides in {slides_split_path}.")

def setup_dataframe_for_one_class(slides: list, c: str, dataset: str) -> pd.DataFrame:
    """Function for setting up dataframe for the slides: filename, label, class and train, val, test (one hot encoded)
    
    Args:
        slides (list): List of slides for one class
        c (str): Class name
        dataset (str): Name of the dataset

    Returns:
        df_slides_one_class (pd.DataFrame): Dataframe for the slides of one class
    """
    df_slides_one_class = pd.DataFrame(slides, columns=["filename"])
    df_slides_one_class["dataset"] = dataset
    df_slides_one_class["label"] = int(Diagnosis(c))
    df_slides_one_class["class"] = c
    df_slides_one_class["train"] = 0
    df_slides_one_class["test"] = 0
    return df_slides_one_class

def get_ids_and_nr_of_duplicates(slides: list) -> np.array:
    """Function to get the patient ids and the number of slides for all patients where multiple WSIs have been made.
    
    Args:
        slides (list): List of slides for one class

    Returns:
        ids_and_number_of_slides (np.array): Array containing the patient ids and the number of slides 
    """
    patient_ids = [slide.split("-")[0] for slide in slides]
    ids_and_number_of_slides = [(id, patient_ids.count(id)) for id in set(patient_ids)]
    # only keep the patient ids where multiple WSIs have been made
    ids_and_number_of_slides = [[id, nr] for id, nr in ids_and_number_of_slides if nr > 1]
    ids_and_number_of_slides = np.array(ids_and_number_of_slides) # such that we can transpose it
    ids_and_number_of_slides = ids_and_number_of_slides.transpose()
    return ids_and_number_of_slides


def mark_slides_with_train_test_split(slides_one_per_patient: list, ids_and_number_of_slides: np.array, 
                                      df_slides_one_class: pd.DataFrame, test_slides_per_class: int):
    """Function to update the one hot encoded columns which split the slides into train and test set.
       Note: Slide names are like '81599-2025-0-HE-CLL' where the 0 stands for 0-th slide of patient 81599.
    
    Args:
        slides_one_per_patient (list): List of slides for one class where only one slide per patient is included
        ids_and_number_of_slides (np.array): Array containing the patient ids and the number of slides for each patient
        df_slides_one_class (pd.DataFrame): Dataframe for the slides of one class
        test_slides_per_class (int): Number of slides for testing
    """
    nr_of_slides = len(slides_one_per_patient)
    for i, filename in enumerate(slides_one_per_patient):
        patient_id = filename.split("-")[0].split("/")[-1] # patient id from CLL/81599-2020-0-HE-CLL
        # check if patient has multiple slides
        if len(ids_and_number_of_slides) == 0:
            nr_of_slides_for_patient = 1
        elif patient_id in ids_and_number_of_slides[0]: # (ids, number of slides)
            index = np.where(ids_and_number_of_slides[0] == patient_id)[0]
            nr_of_slides_for_patient = int(ids_and_number_of_slides[1][index])
        else:
            nr_of_slides_for_patient = 1

        if nr_of_slides_for_patient < 1:
            raise ValueError("Number of slides for patient must be at least 1.")
    
        # depending on the slide, add the slides to the corresponding split
        if i < nr_of_slides - test_slides_per_class:
            mark_slides_of_one_patient_with_split(nr_of_slides_for_patient, filename, df_slides_one_class, "train")
        else:
            # do not use slides from the same patient for testing
            mark_slides_of_one_patient_with_split(1, filename, df_slides_one_class, "test")

def mark_slides_of_one_patient_with_split(nr_of_slides_for_patient: int, filename: str, 
                                          df_slides_one_class: pd.DataFrame, mode: str):
    """Function to mark the slides of one patient with the corresponding split (one-hot-encoding), 
       while ensuring that slides from same patient are in the same split

    Args:
        nr_of_slides_for_patient (int): Number of slides for one patient
        filename (str): Filename of the slide
        df_slides_one_class (pd.DataFrame): Dataframe for the slides of one class
        mode (str): Mode for marking the slides (train, val, test) 
    """
    for j in range(nr_of_slides_for_patient):
        filename = filename.replace("-0-", f"-{j}-")
        df_slides_one_class.loc[df_slides_one_class['filename'] == filename, mode] = 1

####################################################################################################################################
##################################################### Check for data leakage #######################################################
####################################################################################################################################

def check_data_leakage_between_train_and_test_slides(path_to_slides_split: str) -> tuple[set, set]:
    """ Check if the train and test slides are disjoint, i.e. there is no data leakage.
        Returns the train and test slide names as sets. 
    
    Args:
        path_to_slides_split (str): Path to the csv file with the train and test slides

    Returns:
        tuple[set, set]: Train and test slide names as sets
    """
    df = pd.read_csv(path_to_slides_split)
    df_test = df[df["test"] == 1]
    df_train = df[df["train"] == 1]
    train_slide_names = set([name for name in df_train["filename"]])
    test_slide_names = set([name for name in df_test["filename"]])
    print(f"Number of total train slides for cross-validation: {len(train_slide_names)}")
    print(f"Number of test slides for final model evaluation:  {len(test_slide_names)}")
    print()
    if len(test_slide_names.intersection(train_slide_names)) > 0:
        raise ValueError("Train and test slides are not disjoint, i.e. there is data leakage")
    print("Train and test slides are disjoint, i.e. there is no data leakage.")
    return train_slide_names, test_slide_names

def get_target_data_dirs(patch_size: str, dataset: str) -> list:
    """Function to get the target data directories for the specified dataset.

    Args:
        dataset (str): Name of the dataset, e.g. "kiel" or "all_data"

    Returns:
        target_data_dirs (list): List of target data directories
    """
    if dataset == "all_data":
        print("Using all data directories for training (define them if not already done).")
        datasets = ["kiel", "multiply", "swiss_1", "swiss_2"] # just as an example
    else:
        datasets = [dataset]
    target_data_dirs = [os.path.join(patch_size, d, "patches") for d in datasets]
    return target_data_dirs

####################################################################################################################################
############################################################## Main function #######################################################
####################################################################################################################################

def main(patch_size: str, dataset: str, test_slides_per_class: int, shuffle_slides: bool, seed: int, slides_split_name: str):
    """Main function to create the annotation files for training, validation and testing on patch level.
    
    Args:
        patch_size (str): Size of the patches
        dataset (str): Name of the dataset
        test_slides_per_class (int): Number of slides for testing per class
        shuffle_slides (bool): Whether to shuffle the slides
        seed (int): Seed for shuffling the slides
        slides_split_name (str): Name of the csv file where the split of the slides is saved
    """
    # setup
    target_data_dirs = get_target_data_dirs(patch_size, dataset)
    slides_split_path = os.path.join("/data", patch_size, "annotations", dataset, slides_split_name + ".csv")

    # create the split of the classes into train, val and test set if necessary
    if not os.path.exists(slides_split_path):
        os.makedirs(os.path.dirname(slides_split_path), exist_ok=True)
        # compute how many unique slides per class and per dataset should be used for testing 
        path_to_csv = f"/mnt/data_analysis/results/{patch_size}/{dataset}/unique_slides.csv"
        if not os.path.exists(path_to_csv):
            raise ValueError(f"Path {path_to_csv} does not exist. Please make sure to run the script data_analysis/analyze_sampled_data.py first.")
        df = pd.read_csv(path_to_csv, index_col=0)
        df = df.drop("total", axis=0)

        # get number of test slides per class and per data directory
        test_slides_per_class_and_data_dir = get_slides_per_class_per_data_dir(df, test_slides_per_class, target_data_dirs)
        print(f"Number of test slides per class and per data directory:\n{test_slides_per_class_and_data_dir}\n")
        
        print(f"Split of the slides does not exist. Creating it now.")
        create_train_balanced_test_slide_splits(test_slides_per_class_and_data_dir, shuffle_slides, 
                                                seed, slides_split_path)
    else:
        print(f"Split of the slides already exists. Loading it from {slides_split_path} and printing it.")
    df_slides = pd.read_csv(slides_split_path)
    # check if "dataset" column is present, if not add it
    if "dataset" not in df_slides.columns:
        df_slides["dataset"] = dataset
        df_slides.to_csv(slides_split_path, index=False)
        print(f"Added 'dataset' column to the split of the slides and saved it in {slides_split_path}, reloading it.")
        df_slides = pd.read_csv(slides_split_path)

    print(df_slides)

    # check for data leakage
    print("Checking for data leakage.")
    check_data_leakage_between_train_and_test_slides(slides_split_path)



if __name__ == "__main__":
    args = parse_arguments()
    main(**args)