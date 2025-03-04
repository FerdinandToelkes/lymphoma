import os
import pandas as pd


"""
docker run --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code/lymphoma/data_preparation/after_patchcraft:/mnt -v /home/ftoelkes/preprocessed_data/train_data:/data ftoelkes_lymphoma python3 -m ensure_unique_slide_ids
"""
            
def check_if_slide_ids_unique(target_data_dirs: list) -> tuple:
    """This function is designed to be used after running the prepare_data_directory.py script and before running the create_annotations_file.py script. It checks whether the ids of all slides within targt data directories are non overlapping, i.e. due to multiple slides per patient in one data set it is allowed that the slide ids are non unique within one of these sets.
    
    Args:
        target_data_dirs (list): List of strings, each string is the path to a target data directory containing the data_dir directory with the slides.

    Returns:
        tuple: A tuple containing:
            - all_slide_ids (dict): A dictionary with the target data directories as keys and lists of slide ids as values.
            - not_unique_ids (list): A list of slide ids that are not unique across all data sets.
    """   
      
    all_slide_ids = {d: [] for d in target_data_dirs}
    # get all slide ids in the target data directories
    for target_dir in target_data_dirs:
        target_data_dir = os.path.join(target_dir, "data_dir")
        classes = os.listdir(os.path.join("/data", target_data_dir))

        for c in classes:
            path_to_class_dir = os.path.join("/data", target_data_dir, c)
            slides = os.listdir(path_to_class_dir)
            slides = [slide for slide in slides if os.path.isdir(os.path.join(path_to_class_dir, slide))]
            slide_ids = [slide.split("-")[0] for slide in slides]
            all_slide_ids[target_dir].extend(slide_ids) 

    # get rid of duplicates within each data set
    for target_dir in target_data_dirs:
        all_slide_ids[target_dir] = list(set(all_slide_ids[target_dir]))
    
    # check if slide ids are unique across all data sets
    all_slide_ids_flat = [id for ids in all_slide_ids.values() for id in ids]
    if len(all_slide_ids_flat) != len(set(all_slide_ids_flat)):
        print(f"Slide ids are not unique across all data sets, since len(all_slide_ids_flat) = {len(all_slide_ids_flat)} and len(set(all_slide_ids_flat)) = {len(set(all_slide_ids_flat))}")
        not_unique_ids = [item for item, count in pd.Series(all_slide_ids_flat).value_counts().items() if count > 1]
    else:
        print("Slide ids are unique across all specified data sets.")
        not_unique_ids = []

    # print number of unique slides in each data set
    for target_dir in target_data_dirs:
        print(f"Number of unique slides in {target_dir}: {len(all_slide_ids[target_dir])}")
    return all_slide_ids, not_unique_ids

def permute_not_unique_ids(target_dir: str, not_unique_ids: list):
    """This function permutes the slide ids of the slides that are not unique across all data sets. It generates a new slide id by permuting the old slide id, e.g. 12345 -> 23451. It renames the directories of the slides accordingly and saves the old to new slide ids mapping in a csv file.

    Args:
        target_dir (str): The path to the target data directory containing the data_dir directory with the slides.
        not_unique_ids (list): A list of slide ids that are not unique across all data sets.
    """
    target_data_dir = os.path.join(target_dir, "data_dir")
    path_to_csv = f"/data/{target_dir}/old_to_new_slide_ids.csv"
    if os.path.exists(path_to_csv):
        print(f"old_to_new_slide_ids.csv already exists at {path_to_csv}, returning None.")
        return
    classes = os.listdir(os.path.join("/data", target_data_dir))
    not_unique_slides = []
    for c in classes:
        path_to_class_dir = os.path.join("/data", target_data_dir, c)
        slides = os.listdir(path_to_class_dir)
        slides = [slide for slide in slides if os.path.isdir(os.path.join(path_to_class_dir, slide))]
        not_unique_slides.extend([slide for slide in slides if slide.split("-")[0] in not_unique_ids])
    
    print(f"not_unique_slides: {not_unique_slides}")
    # obtain the slide ids of the not unique slides, can have multiple slides per patient
    not_unique_slide_ids = list(set([slide.split("-")[0] for slide in not_unique_slides]))
    # check if all not unique slides were found in the specified target data directory
    if len(not_unique_slide_ids) != len(not_unique_ids):
        print(f"not all not unique slides were found in {target_dir}")
        return
    
    old_to_new_slide = {old: old for old in not_unique_slides}
    for name in not_unique_slides:
        # generate new name for the slide
        slide_id = name.split("-")[0]
        slide_id_list = list(slide_id)
        # permute the slide id, e.g. 12345 -> 23451
        new_slide_id = slide_id_list[1:] + slide_id_list[:1]
        new_slide_id = "".join(new_slide_id)
        new_name = name.replace(slide_id, new_slide_id)
        old_to_new_slide[name] = new_name

        # rename the directories
        path_to_class_dir = os.path.join("/data", target_data_dir, name.split("-")[-1])
        path_to_slide = os.path.join(path_to_class_dir, name)
        path_to_new_slide = os.path.join(path_to_class_dir, new_name)
        os.rename(path_to_slide, path_to_new_slide)
        print(f"Renamed {path_to_slide} to {path_to_new_slide}")
    
    # save the old to new slide ids mapping 
    old_to_new_slide_df = pd.DataFrame(old_to_new_slide.items(), columns=["old_slide_id", "new_slide_id"])
    old_to_new_slide_df.to_csv(path_to_csv, index=False)

    


if __name__ == "__main__":
    datasets = ["kiel"]
    target_data_dirs = [os.path.join("1024um", d, "patches") for d in datasets]
    all_slide_ids, not_unique_ids = check_if_slide_ids_unique(target_data_dirs)

    if len(not_unique_ids) > 0:
        print(f"not_unique_ids: {not_unique_ids}")
        print("Checking if permuting slide ids makes them unique...")
        permute_not_unique_ids(target_data_dirs[-1], not_unique_ids)

        # check if slides are unique after permutation
        all_slide_ids, not_unique_ids = check_if_slide_ids_unique(target_data_dirs)
        if len(not_unique_ids) > 0:
            print(f"not_unique_ids: {not_unique_ids}")
            print("Permutation of slide ids did not make them unique.")
        else:
            print("Permutation of slide ids made them unique.")
    print("Done.")