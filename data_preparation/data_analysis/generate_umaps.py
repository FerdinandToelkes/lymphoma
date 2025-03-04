import torch
from umap import UMAP

import argparse
import numpy as np
import os
import pickle



"""
screen -dmS generate_umap sh -c 'docker run --shm-size=400gb --gpus all --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code/lymphoma/data_preparation/data_analysis:/mnt -v /home/ftoelkes/preprocessed_data/test_data:/data ftoelkes_lymphoma python3 -m generate_umaps --patch_size=1024um --data_dir=embeddings_dir --dataset=kiel --data_specifier=patches --classes=CLL,DLBCL,FL,HL,LTDS,MCL --unique_slides; exec bash'
"""

def parse_arguments() -> dict:
    """ Parse command line arguments. 
    
    Returns:
        dict: Dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='UMAP generation.')
    # parameters needed to specify all the different paths
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('--data_specifier', default='patches', type=str, help='String specifying the patches to be used, e.g top_5_patches (default: "patches")')
    parser.add_argument('-ds', '--datasets', default='kiel', type=str, help='Name of the datasets seperated by commas, can be a combination of "kiel", "swiss_1", "swiss_2", "multiply", "munich" (default: kiel)')
    parser.add_argument("--data_dir", default="embeddings_dir", type=str, help="Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'. (default: embeddings_dir)")
    parser.add_argument('--classes', default='CLL,DLBCL,FL,HL,LTDS,MCL', type=str, help='Classes to be used for the analysis seperated by commas. (default: "CLL,DLBCL,FL,HL,LTDS,MCL")')
    parser.add_argument('--unique_slides', default=True, action='store_true', help='Whether to only take the zeroth slide of a patient, i.e. 80212-2018-0-HE-FL but not 80212-2018-1-HE-FL (default: False)')
    # parameters that will likely not be changed
    parser.add_argument('--n_neighbors', default=15, type=int, help='Number of neighbors to use for UMAP. (default: 15)')
    parser.add_argument('--min_dist', default=0.1, type=float, help='Minimum distance to use for UMAP. (default: 0.1)')
    parser.add_argument('-s', '--seed', default=42, type=int, help='Random seed (default: 42)')
    return dict(vars(parser.parse_args())) 

def parse_classes(classes: str) -> list[str]:
    """Parse the classes from the string.

    Args:
        classes (str): String containing the classes separated by commas.

    Returns:
        List of classes.
    """
    classes = classes.split(",")
    classes.sort()
    return classes

def parse_datasets(datasets: str) -> list[str]:
    """Parse the datasets from the string.

    Args:
        datasets (str): String containing the datasets separated by commas.

    Returns:
        List of datasets.
    """
    datasets = datasets.split(",")
    datasets.sort()
    return datasets


def collect_embeddings(patch_size: str, data_specifier: str, data_dir: str, 
                       classes: list[str], datasets: list[str], unique_slides: bool = True) -> dict:
    """Collect all embeddings for all the desired classes from the given path to the data.

    Args:
        patch_size (str): Size of the patches.
        data_specifier (str): String specifying the patches to be used, e.g top_5_patches.
        data_dir (str): Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'.
        classes (list[str]): List of classes to collect the embeddings for.
        datasets (list[str]): List of datasets to collect the embeddings for.
        unique_slides (bool): Whether to only take the zeroth slide of a patient, i.e. 01072-2025-0-HE-FL but not 01072-2025-1-HE-FL.

    Returns:
        Dictionary containing the embeddings for each class as a dictionary with the slide names 
        as keys and the embeddings as concatenated tensors as values.
    """
    all_embeddings = {}
    len_datasets = len(datasets)
    for dataset in datasets:
        path_to_data = f"/data/{patch_size}/{dataset}/{data_specifier}/{data_dir}"
        for c in classes:
            class_embeddings = {}
            class_dir = os.path.join(path_to_data, c)
            slides = os.listdir(class_dir)
            slides = [s for s in slides if os.path.isdir(os.path.join(class_dir, s))]
            if unique_slides:
                # only take the zeroth slide of a patient, i.e. 01072-2025-0-HE-FL but not 01072-2025-1-HE-FL
                slides = [s for s in slides if s.split("-")[2] == "0"]
            nr_of_slides = len(slides)
            if nr_of_slides == 0:
                print(f"No slides found for class {c}.")
                continue
            for i,s in enumerate(slides):
                print(f"Processing slide {i+1}/{nr_of_slides} for class {c}.", end="\r")
                class_embeddings[s] = get_embeddings_for_slide(class_dir, s)

            print(f"For class {c} in dataset {dataset} embeddings from {len(class_embeddings)} slides were taken.")
            if len_datasets == 1:
                all_embeddings[c] = class_embeddings
            else:
                all_embeddings[f"{dataset}_{c}"] = class_embeddings
    return all_embeddings

def get_embeddings_for_slide(class_dir: str, slide: str) -> list[torch.Tensor]:
    """Get the embeddings for a slide.

    Args:
        class_dir (str): Path to the class directory.
        slide (str): Name of the slide.

    Returns:
        List of embeddings for the slide.
    """
    slide_dir = os.path.join(class_dir, slide)
    patches = os.listdir(slide_dir)
    patches = [p for p in patches if p.startswith("patch") and p.endswith(".pt")]
    embs_one_slide = []
    for p in patches:
        emb, label = torch.load(os.path.join(slide_dir, p))
        embs_one_slide.append(emb)
    return embs_one_slide

def put_all_embeddings_together(all_embeddings_dict: dict) -> np.ndarray:
    """Put all embeddings together in one numpy array.

    Args:
        all_embeddings_dict (dict): Dictionary containing the embeddings for each class as a dictionary with the slide names.

    Returns:
        Numpy array containing all embeddings.
    """
    only_embeddings = []
    for slides_dict in all_embeddings_dict.values():
        for slide_embs in slides_dict.values():
            only_embeddings.extend(slide_embs)
    embeddings_np = torch.stack(only_embeddings).numpy()
    return embeddings_np

def split_reduced_embeddings(reduced_embeddings: np.ndarray, all_embeddings_dict: dict) -> dict:
    """Split the reduced embeddings back into the classes and slides.

    Args:
        reduced_embeddings (np.ndarray): Reduced embeddings.
        all_embeddings_dict (dict): Dictionary containing the embeddings for each class.

    Returns:
        Dictionary containing the reduced embeddings for each class as a dictionary with the slide names.
    """
    reduced_embeddings_dict = {}
    classes = list(all_embeddings_dict.keys())
    start = 0
    for c in classes:
        reduced_embeddings_dict[c] = {}
        slides = all_embeddings_dict[c].keys()
        start = 0
        for s in slides:
            end = start + len(all_embeddings_dict[c][s])
            reduced_embeddings_dict[c][s] = reduced_embeddings[start:end]
            start = end
    # check whether this dict has the same basic shape as the original dict
    check_shapes(all_embeddings_dict, reduced_embeddings_dict)
    return reduced_embeddings_dict

def check_shapes(all_embeddings_dict: dict, reduced_embeddings_dict: dict):
    """Check whether the general forms of the dictionarys match.

    Args:
        all_embeddings_dict (dict): Dictionary containing the embeddings for each class.
        reduced_embeddings_dict (dict): Dictionary containing the reduced embeddings for each class.
    """
    for c in all_embeddings_dict.keys():
        if len(all_embeddings_dict[c]) != len(reduced_embeddings_dict[c]):
            raise ValueError(f"Number of slides in class {c} does not match.")
        for s in all_embeddings_dict[c].keys():
            if len(all_embeddings_dict[c][s]) != len(reduced_embeddings_dict[c][s]):
                raise ValueError(f"Number of embeddings in slide {s} does not match.")

def save_reduced_embeddings(reduced_embeddings_dict: dict, path: str, classes: str, unique_slides: bool, n_neighbors: int, min_dist: float):
    """Save the reduced embeddings to file as pickle.

    Args:
        reduced_embeddings_dict (dict): Dictionary containing the reduced embeddings for each class.
        path (str): Path to save the reduced embeddings to.
        classes (str): String containing the classes separated by commas.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    name = f"reduced_embeddings_{classes}_us={unique_slides}_nn={n_neighbors}_md={min_dist}.pkl"
    with open(f"{path}/{name}", "wb") as f:
        pickle.dump(reduced_embeddings_dict, f)

def main(patch_size: str, data_specifier: str, datasets: str, data_dir: str, classes: str, 
         unique_slides: bool, n_neighbors: int, min_dist: float, seed: int):
    """Main function to generate UMAP plots for the embeddings.

    Args:
        patch_size (str): Size of the patches.
        data_specifier (str): String specifying the patches to be used, e.g top_5_patches.
        datasets (str): List of datasets to collect the embeddings for.
        data_dir (str): Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'.
        classes (str): List of classes to collect the embeddings for.
        unique_slides (bool): Whether to only take the zeroth slide of a patient, i.e. 01072-2025-0-HE-FL but not 01072-2025-1-HE-FL.
        n_neighbors (int): Number of neighbors to use for UMAP.
        min_dist (float): Minimum distance to use for UMAP.
        seed (int): Random seed.
    """
    # Path to the data class dirs
    classes_list = parse_classes(classes)
    datasets = parse_datasets(datasets)

    all_embeddings_dict = collect_embeddings(patch_size, data_specifier, data_dir, classes_list, datasets, unique_slides)
    print(f"All embeddings collected for {all_embeddings_dict.keys()}")
        
    # put all embeddings together
    embeddings_np = put_all_embeddings_together(all_embeddings_dict)
    print(f"Shape of all embeddings put together: {embeddings_np.shape}")

    # get reduced embeddings using UMAP
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    reduced_embeddings = umap.fit_transform(embeddings_np)
    
    # split the reduced embeddings back into the classes
    reduced_embeddings_dict = split_reduced_embeddings(reduced_embeddings, all_embeddings_dict)

    # save the reduced embeddings to file
    if len(datasets) == 1:
        dataset_name = datasets[0]
    else:
        dataset_name = "_".join(datasets)
    path = f"./results/{patch_size}/{dataset_name}/umap_plots"
    save_reduced_embeddings(reduced_embeddings_dict, path, classes, unique_slides, n_neighbors, min_dist)
   


if __name__ == '__main__':
    args = parse_arguments()
    main(**args)