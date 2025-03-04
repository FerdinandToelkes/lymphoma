import os
import torch
import pandas as pd
import argparse
import numpy as np
import pickle

import torchvision.transforms.v2 as T 
from tqdm import tqdm

from data_preparation.embedding_generation.gen_and_concat_top_embs import get_max_indices, get_max_indices_without_overlap


# NOTE: Make sure that the functions match the ones used in gen_and_concat_top_embs.py


"""
screen -dmS analyze_data0 sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code/lymphoma:/mnt -v /home/ftoelkes/preprocessed_data/test_data:/data -v /home/ftoelkes/code/lymphoma/vit_large_patch16_256.dinov2.uni_mass100k:/model ftoelkes_lymphoma python3 -m uni_analysis.analyze_data --nr_of_compared_max_vals=1 ; docker run --shm-size=400gb --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code/lymphoma:/mnt -v /home/ftoelkes/preprocessed_data/test_data:/data -v /home/ftoelkes/code/lymphoma/vit_large_patch16_256.dinov2.uni_mass100k:/model ftoelkes_lymphoma python3 -m uni_analysis.analyze_data --nr_of_compared_max_vals=2 ; docker run --shm-size=400gb --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code/lymphoma:/mnt -v /home/ftoelkes/preprocessed_data/test_data:/data -v /home/ftoelkes/code/lymphoma/vit_large_patch16_256.dinov2.uni_mass100k:/model ftoelkes_lymphoma python3 -m uni_analysis.analyze_data --nr_of_compared_max_vals=3 ; docker run --shm-size=400gb --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code/lymphoma:/mnt -v /home/ftoelkes/preprocessed_data/test_data:/data -v /home/ftoelkes/code/lymphoma/vit_large_patch16_256.dinov2.uni_mass100k:/model ftoelkes_lymphoma python3 -m uni_analysis.analyze_data --nr_of_compared_max_vals=4 ; docker run --shm-size=400gb --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code/lymphoma:/mnt -v /home/ftoelkes/preprocessed_data/test_data:/data -v /home/ftoelkes/code/lymphoma/vit_large_patch16_256.dinov2.uni_mass100k:/model ftoelkes_lymphoma python3 -m uni_analysis.analyze_data --nr_of_compared_max_vals=5 ; exec bash'
"""


def parse_arguments() -> dict:
    """ Parse the arguments for the analysis. 
    
    Returns:
        dict: Dictionary with the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Analyse differences between embeddings and attention maps of differently preprocessed data.')
    # general arguments
    parser.add_argument('--target_dir', default="test_embeddings_dir", type=str, help='Directory to save the embeddings and attentions to. (default: test_embeddings_dir)')
    parser.add_argument('-ps', '--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply" or "munich". (default: kiel)')
    parser.add_argument('-os', '--original_size', default=4096, type=int, help='Original size of the patches. (default: 4096)')
    # for the attention analysis
    parser.add_argument('--nr_of_compared_max_vals', default=3, type=int, help='Number of maximal attention values to compare between each map. (default: 3)')
    parser.add_argument('-ao', '--allow_overlap', default=False, action='store_true', help='Allow overlapping regions in the top n attended regions. (default: False, if flag is set: True)')
    args = parser.parse_args()
    return dict(vars(args))

##################################################################################################################################
################################################## Embedding analysis functions ##################################################
##################################################################################################################################

def setup_embedding_analysis(target_dir: str, patch_size: str, dataset: str) -> tuple[list]:
    """ Setup the paths to the embeddings and the result directory. 
    
    Args:
        target_dir (str): The target directory where the embeddings are stored.
        patch_size (str): The size of the patches.
        dataset (str): The name of the dataset.

    Returns:
        tuple[list]: List of paths to the patches, paths to the embedding directories, the variants and the result directory.
    """
    path_to_test_embeddings_dir = os.path.join("/data", patch_size, dataset, "patches", target_dir)
    variants = os.listdir(path_to_test_embeddings_dir)
    # only take the directories
    variants = [var for var in variants if os.path.isdir(os.path.join(path_to_test_embeddings_dir, var))]
    paths_to_embedding_dirs = [os.path.join(path_to_test_embeddings_dir, variant) for variant in variants]
    paths_to_patches = get_paths_to_embeddings(paths_to_embedding_dirs[0])
    result_dir = "./uni_analysis/results"
    os.makedirs(result_dir, exist_ok=True)
    return paths_to_patches, paths_to_embedding_dirs, variants, result_dir


def get_paths_to_embeddings(paths_to_org_embedding: list) -> list:
    """ Get the paths to the embeddings of the patches. 
    
    Args:
        paths_to_org_embedding (list): List of paths to the original embeddings.

    Returns:
        list: List of paths to the patches.
    """
    paths_to_patches = []
    class_dirs = os.listdir(paths_to_org_embedding)
    for c in class_dirs:
        slides = os.listdir(os.path.join(paths_to_org_embedding, c))
        for slide in slides:
            patches = os.listdir(os.path.join(paths_to_org_embedding, c, slide))
            paths_to_patches.extend([os.path.join(c, slide, patch) for patch in patches])
    # filter out non-patch files
    paths_to_patches = [p for p in paths_to_patches if p.endswith(".pt") and not p.endswith("_attentions.pt")]

    print(f"Number of patches: {len(paths_to_patches)}")
    return paths_to_patches

def get_embedding_statistics(original_size: int, paths_to_patches: list, 
                             paths_to_embedding_dirs: list, variants: list) -> dict:
    """Compute the cosine similarities and mean squared errors between the embeddings 
    of the different variants, their mean, std and the resulting standard error.
    
    Args:
        original_size (int): The original size of the patches.
        paths_to_patches (list): List of paths to the patches.
        paths_to_embedding_dirs (list): List of paths to the embedding directories.
        variants (list): List of the variants.

    Returns:
        dict: Dictionary with the cosine similarities and mean squared errors.
    """
    # setup the data container
    cosine_similarities = {var: {"values": [], "mean": 0, "std": 0, "std_err": 0} for var in variants if var != f"{original_size}"}
    mean_squared_errors = {var: {"values": [], "mean": 0, "std": 0, "std_err": 0} for var in variants if var != f"{original_size}"}
    # compute the cosine similarities and mean squared errors
    cosine_similarities, mean_squared_errors = get_single_embedding_statistics(original_size, paths_to_patches, paths_to_embedding_dirs, 
                                                                                   variants, cosine_similarities, mean_squared_errors)
    # compute the averages, stds and std errors
    cosine_similarities, mean_squared_errors = get_mean_std_and_error_for_emb_stats(cosine_similarities, mean_squared_errors)
    return cosine_similarities, mean_squared_errors


def get_single_embedding_statistics(original_size: int, paths_to_patches: list, paths_to_embedding_dirs: list, 
                                        variants: list, cosine_similarities: dict, mean_squared_errors: dict) -> tuple[dict]:
    """ Compute the cosine similarity and mean squared error for the embeddings of the different variants. 
    
    Args:
        original_size (int): The original size of the patches.
        paths_to_patches (list): List of paths to the patches.
        paths_to_embedding_dirs (list): List of paths to the embedding directories.
        variants (list): List of the variants.
        cosine_similarities (dict): Dictionary with the cosine similarities.
        mean_squared_errors (dict): Dictionary with the mean squared errors.

    Returns:
        tuple[dict]: Tuple with the updated cosine similarities and mean squared errors.
    """
    for path in paths_to_patches:
        # load embeddings (and labels)
        embeddings = {var: None for var in variants}
        for var in variants:
            embeddings[var], _ = torch.load(os.path.join(paths_to_embedding_dirs[variants.index(var)], path))
        # compute the difference, cosine sim and mse to the original embedding
        for var in cosine_similarities.keys():
            cos_sim = torch.nn.functional.cosine_similarity(embeddings[var], embeddings[f"{original_size}"], dim=0).item()
            mse = torch.nn.functional.mse_loss(embeddings[var], embeddings[f"{original_size}"]).item()
            cosine_similarities[var]["values"].append(cos_sim)
            mean_squared_errors[var]["values"].append(mse)
    return cosine_similarities, mean_squared_errors

def get_mean_std_and_error_for_emb_stats(cosine_similarities: dict, mean_squared_errors: dict) -> tuple[dict]:
    """ Compute the mean, std and standard error for the cosine similarities and mean squared errors. 
    
    Args:
        cosine_similarities (dict): Dictionary with the cosine similarities.
        mean_squared_errors (dict): Dictionary with the mean squared errors.

    Returns:
        tuple[dict]: Tuple with the updated cosine similarities and mean squared errors.
    """
    for var in cosine_similarities.keys():
        cosine_similarities[var]["mean"] = np.mean(cosine_similarities[var]["values"])
        cosine_similarities[var]["std"] = np.std(cosine_similarities[var]["values"])
        mean_squared_errors[var]["mean"] = np.mean(mean_squared_errors[var]["values"])
        mean_squared_errors[var]["std"] = np.std(mean_squared_errors[var]["values"])
        cosine_similarities[var]["std_err"] = cosine_similarities[var]["std"] / np.sqrt(len(cosine_similarities[var]["values"]))
        mean_squared_errors[var]["std_err"] = mean_squared_errors[var]["std"] / np.sqrt(len(mean_squared_errors[var]["values"]))
    return cosine_similarities, mean_squared_errors
    
def save_full_embedding_statistics(cosine_similarities: dict, mean_squared_errors: dict, path: str):
    """ Save the single values, means, standard deviations and standard errors as pickle file. 
    
    Args:
        cosine_similarities (dict): Dictionary with the cosine similarities.
        mean_squared_errors (dict): Dictionary with the mean squared errors.
        path (str): Path to the result directory.
    """
    stats = {"cosine_similarities": cosine_similarities, "mean_squared_errors": mean_squared_errors}
    path = os.path.join(path, "full_embedding_statistics.pkl")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
    with open(path, "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved the full embedding statistics to {path}")

   
def save_embedding_statistics(cosine_similarities: dict, mean_squared_errors: dict, path: str):
    """ Save the statistics as pandas dataframe to a csv file. 
    
    Args:
        cosine_similarities (dict): Dictionary with the cosine similarities.
        mean_squared_errors (dict): Dictionary with the mean squared errors.
        path (str): Path to the result directory.
    """
    path = os.path.join(path, "embedding_statistics.csv")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
    
    data = []
    cosine_keys = cosine_similarities.keys()
    cosine_keys = sorted(cosine_keys, key=lambda x: int(x.split("_")[0]), reverse=True)
    for res in cosine_keys:
        data.append([res, cosine_similarities[res]["mean"], cosine_similarities[res]["std"], cosine_similarities[res]["std_err"],
                     mean_squared_errors[res]["mean"], mean_squared_errors[res]["std"], mean_squared_errors[res]["std_err"]])
    df = pd.DataFrame(data, columns=["Variation", "Cosine Similarity Avg", "Cosine Similarity Std", "Cosine Similarity Std Error",
                                     "Mean Squared Error Avg", "Mean Squared Error Std", "Mean Squared Error Std Error"])
    df.to_csv(path, index=False)
    print(f"Saved the embedding statistics to {path}")



######################################################################################################################################
################################################## Attention map analysis functions ##################################################
######################################################################################################################################

# if baseline width is too large one can consider fixing it such that the total overlap is fixed
BASELINE_WIDTH = 400 # this correpsonds to 100um patches being sampled at full resolution
def get_attention_statistics(original_size: int, paths_to_patches: list, paths_to_embedding_dirs: list, 
                                variants: list, nr_of_compared_max_vals: int, allow_overlap: bool) -> tuple[dict]:
    """Load attentions, prepare them, extract the indices of maximimal values and compute the mean squared error, the indices distance and the overlap of the top regions for the augmented attentions. 
    
    Args:
        original_size (int): The original size of the patches.
        paths_to_patches (list): List of paths to the patches.
        paths_to_embedding_dirs (list): List of paths to the embedding directories.
        variants (list): List of the variants, i.e. the different augmentations and resizing.
        nr_of_compared_max_vals (int): Number of maximal attention values to compare between each map.
        allow_overlap (bool): Allow overlapping regions in the top n attended regions.

    Returns:
        tuple[dict]: Tuple with the mean squared errors, indices distances and top regions overlap.
    """
    mean_squared_errors = {var: {"values": [], "mean": 0, "std": 0, "std_err": 0} for var in variants if var != f"{original_size}_attentions"}
    indices_distances = {var: {"values": [], "mean": 0, "std": 0, "std_err": 0} for var in variants if var != f"{original_size}_attentions"}
    top_regions_overlap = {var: {"values": [], "mean": 0, "std": 0, "std_err": 0} for var in variants if var != f"{original_size}_attentions"}

    for path in tqdm(paths_to_patches, total=len(paths_to_patches), desc="Computing attention statistics"):
        # load attentions and normalize them
        path = path.split(".pt")[0] + "_attentions.pt"
        attentions = {var: None for var in variants}
        top_indices = {var: None for var in variants}
        top_attn_regions = {var: None for var in variants}
        for var in variants:
            # load attn and average over different heads (each head is already normalized)
            resized_attn = load_and_prepare_attn(paths_to_embedding_dirs, path, variants, var, original_size)
            # note that pytorch expects a channel dimension -> unsqueeze
            attentions[var] = resized_attn
            
            # Note: passing the resized attn leads to better results since the indices are 
            # "more accurate" when comparing the overlap of the top regions
            if allow_overlap:
                current_top_indices = get_max_indices(resized_attn, nr_of_compared_max_vals, original_size)
            else:
                current_top_indices = get_max_indices_without_overlap(resized_attn, nr_of_compared_max_vals, BASELINE_WIDTH, original_size)
            top_indices[var] = current_top_indices
            # make a mask with ones at sqaures around the top indices
            mask_top_regs = get_mask_at_indices(original_size, top_indices[var], width_square=BASELINE_WIDTH)  
            top_attn_regions[var] = mask_top_regs

        # compute the mse to the original attention map (no augmentation)
        mean_squared_errors, indices_distances, top_regions_overlap = get_single_attn_statistics(original_size, attentions, top_indices, top_attn_regions, mean_squared_errors, indices_distances, top_regions_overlap, nr_of_compared_max_vals)

    # compute the averages, stds and std errors
    mean_squared_errors, indices_distances, top_regions_overlap = get_mean_std_and_err_attn_stats(mean_squared_errors, indices_distances, top_regions_overlap)
    return mean_squared_errors, indices_distances, top_regions_overlap

def load_and_prepare_attn(paths_to_embedding_dirs: list, path: str, variants: list, var: str, 
                          original_size: int) -> torch.Tensor:
    """ Load the attention map and prepare it for comparison. 
    
    Args:
        paths_to_embedding_dirs (list): List of paths to the embedding directories.
        path (str): Path to the attention map.
        variants (list): List of the variants.
        var (str): The current variant.
        original_size (int): The original size of the patches.

    Returns:
        torch.Tensor: The resized and normalized attention map.
    """
    attn = torch.load(os.path.join(paths_to_embedding_dirs[variants.index(var)], path))
    attn = attn.mean(0)
    # resize attention to original size and normalize it
    resized_attn = T.Resize((original_size, original_size), antialias=True)(attn.unsqueeze(0)).squeeze()
    resized_attn = (resized_attn - resized_attn.min()) / (resized_attn.max() - resized_attn.min())
    # reverse the flip operations -> needed for fair comparison
    if "hflip" in var:
        resized_attn = resized_attn.flip(0)
    if "vflip" in var:
        resized_attn = resized_attn.flip(1)
    return resized_attn

# compare with figure in thesis and get_patches_at_indices in concatenate_top_embeddings.py
def get_mask_at_indices(original_size: int, top_indices: list, width_square: int) -> np.ndarray:
    """ Create a mask with ones at squares around the top indices and push them inside the 
    image if they are outside of the original image. 
    
    Args:
        original_size (int): The original size of the patches.
        top_indices (list): List of the top indices.
        width_square (int): The width of the square.

    Returns:
        np.ndarray: The mask with ones at the squares around the top indices.
    """
    mask_top_regs = np.zeros((original_size, original_size))
    half_width = width_square // 2
    for ind in top_indices:
        # if positions lead to squares outside the image, push them inside
        x_min = ind[0] - half_width
        y_min = ind[1] - half_width
        x_max = ind[0] + half_width
        y_max = ind[1] + half_width
        if x_min < 0:
            x_max -= x_min # add the amount it is outside
            x_min = 0
        elif x_max > original_size:
            x_min -= x_max - original_size # subtract the amount it is outside
            x_max = original_size   
        if y_min < 0:
            y_max -= y_min # add the amount it is outside
            y_min = 0
        elif y_max > original_size:
            y_min -= y_max - original_size # subtract the amount it is outside
            y_max = original_size
        mask_top_regs[x_min:x_max, y_min:y_max] = 1
    return mask_top_regs

def get_single_attn_statistics(original_size: int, attentions: dict, top_indices: dict, top_attn_regions: dict,
                                mean_squared_errors: dict, indices_distances: dict, top_regions_overlap: dict,
                                nr_of_compared_max_vals: int) -> tuple[dict]:
    """ Compute the mean squared error, the indices distance and the overlap of 
        the top regions between different attentions. 

    Args:
        original_size (int): The original size of the patches.
        attentions (dict): Dictionary with the attention maps.
        top_indices (dict): Dictionary with the top indices.
        top_attn_regions (dict): Dictionary with the top attention regions.
        mean_squared_errors (dict): Dictionary with the mean squared errors.
        indices_distances (dict): Dictionary with the indices distances.
        top_regions_overlap (dict): Dictionary with the top regions overlap.
        nr_of_compared_max_vals (int): Number of maximal attention values to compare between each map.

    Returns:
        tuple[dict]: Tuple with the updated mean squared errors, indices distances and top regions overlap.    
    """
    # compute the mean squared error to the original attention map   
    for var in mean_squared_errors.keys():
        mse = torch.nn.functional.mse_loss(attentions[var], attentions[f"{original_size}_attentions"]).item()
        mean_squared_errors[var]["values"].append(mse)
        # compute the overlap of the top regions
        overlap = np.sum(top_attn_regions[var] * top_attn_regions[f"{original_size}_attentions"])     
        # scale factor is not neccessarily the same for all variants due to overlapping regions
        scale_factor = max(np.sum(top_attn_regions[f"{original_size}_attentions"]), np.sum(top_attn_regions[var]))
        overlap = overlap / scale_factor
        top_regions_overlap[var]["values"].append(overlap)
    
    if nr_of_compared_max_vals == 1:
        return mean_squared_errors, indices_distances, top_regions_overlap

    # compute distances between the top indices within the same variant
    for var in mean_squared_errors.keys():
        np_top_indices = np.array(top_indices[var])
        # not perfect but not too inefficient
        pairwise_distances = np.array([np.linalg.norm(np_top_indices[i+1] - np_top_indices[i]) for i in range(len(np_top_indices)-1)])
        ind_dist = np.mean(pairwise_distances)
        indices_distances[var]["values"].append(ind_dist)
    return mean_squared_errors, indices_distances, top_regions_overlap

def get_mean_std_and_err_attn_stats(mean_squared_errors: dict, indices_distances: dict, 
                                    top_regions_overlap: dict) -> tuple[dict]:
    """ Compute the mean, std and std error for the mean squared errors, indices 
        distances and top regions overlap. 
    
    Args:
        mean_squared_errors (dict): Dictionary with the mean squared errors.
        indices_distances (dict): Dictionary with the indices distances.
        top_regions_overlap (dict): Dictionary with the top regions overlap.

    Returns:
        tuple[dict]: Tuple with the updated mean squared errors, indices distances and top regions overlap.
    """
    for var in mean_squared_errors.keys():
        mean_squared_errors[var]["mean"] = np.mean(mean_squared_errors[var]["values"])
        mean_squared_errors[var]["std"] = np.std(mean_squared_errors[var]["values"])
        mean_squared_errors[var]["std_err"] = mean_squared_errors[var]["std"] / np.sqrt(len(mean_squared_errors[var]["values"]))
        top_regions_overlap[var]["mean"] = np.mean(top_regions_overlap[var]["values"])
        top_regions_overlap[var]["std"] = np.std(top_regions_overlap[var]["values"])
        top_regions_overlap[var]["std_err"] = top_regions_overlap[var]["std"] / np.sqrt(len(top_regions_overlap[var]["values"]))
        if len(indices_distances[var]["values"]) > 0:
            indices_distances[var]["mean"] = np.mean(indices_distances[var]["values"])
            indices_distances[var]["std"] = np.std(indices_distances[var]["values"])
            indices_distances[var]["std_err"] = indices_distances[var]["std"] / np.sqrt(len(indices_distances[var]["values"]))
    return mean_squared_errors, indices_distances, top_regions_overlap

def save_full_attention_statistics(mean_squared_errors: dict, indices_distances: dict, 
                                top_regions_overlap: dict, path: str, nr_of_compared_max_vals: int):
    """ Save the single values, means, standard deviations and standard errors as pickle file. 
    
    Args:
        mean_squared_errors (dict): Dictionary with the mean squared errors.
        indices_distances (dict): Dictionary with the indices distances.
        top_regions_overlap (dict): Dictionary with the top regions overlap.
        path (str): Path to the result directory.
        nr_of_compared_max_vals (int): Number of maximal attention values to compare between each map.

    Returns:
        tuple[dict]: Tuple with the updated mean squared errors, indices distances and top regions overlap
    """
    stats = {"mean_squared_errors": mean_squared_errors, "indices_distances": indices_distances,
             "top_regions_overlap": top_regions_overlap}
    path = os.path.join(path, f"full_attn_stats_{nr_of_compared_max_vals}_compared_regions.pkl")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
    with open(path, "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved the full embedding statistics to {path}")

   
def save_attention_statistcs(mean_squared_errors: dict, indices_distances: dict, 
                        top_regions_overlap: dict, path: str, nr_of_compared_max_vals: int):
    """ Save the statistics as pandas dataframe to a csv file. 
    
    Args:
        mean_squared_errors (dict): Dictionary with the mean squared errors.
        indices_distances (dict): Dictionary with the indices distances.
        top_regions_overlap (dict): Dictionary with the top regions overlap.
        path (str): Path to the result directory.
        nr_of_compared_max_vals (int): Number of maximal attention values to compare between each
    """
    path = os.path.join(path, f"attn_stats_{nr_of_compared_max_vals}_compared_regions.csv")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
    
    data = []
    mse_keys = mean_squared_errors.keys()
    mse_keys = sorted(mse_keys, key=lambda x: int(x.split("_")[0]), reverse=True)
    for res in mse_keys:
        data.append([res, mean_squared_errors[res]["mean"], mean_squared_errors[res]["std"], mean_squared_errors[res]["std_err"], 
                     indices_distances[res]["mean"], indices_distances[res]["std"], indices_distances[res]["std_err"],
                     top_regions_overlap[res]["mean"], top_regions_overlap[res]["std"], top_regions_overlap[res]["std_err"]])  
    df = pd.DataFrame(data, columns=["Variation", "Mean Squared Error Avg", "Mean Squared Error Std", "Mean Squared Error Std Error",
                                     "Indices Distance Avg", "Indices Distance Std", "Indices Distance Std Error",
                                     "Top Regions Overlap Avg", "Top Regions Overlap Std", "Top Regions Overlap Std Error"])
    df.to_csv(path, index=False)
    print(f"Saved the embedding statistics to {path}")



###############################################################################################################################


def main(target_dir: str, patch_size: str, dataset: str, original_size: int, 
         nr_of_compared_max_vals: int, allow_overlap: bool):
    """ Main function for the analysis of the embeddings and attention maps.

    Args:
        target_dir (str): The target directory where the embeddings are stored.
        patch_size (str): The size of the patches.
        dataset (str): The name of the dataset.
        original_size (int): The original size of the patches.
        nr_of_compared_max_vals (int): Number of maximal attention values to compare between each map.
        allow_overlap (bool): Allow overlapping regions in the top n attended regions.
    """
    # setup for the embedding analysis
    print(f"Analyzing embeddings and attentions for {nr_of_compared_max_vals} compared regions.")
    paths_to_patches, paths_to_embedding_dirs, variants, result_dir = setup_embedding_analysis(target_dir, patch_size, dataset)

    # compute the cosine similarities and mean squared errors and save them 
    cosine_similarities, mean_squared_errors = get_embedding_statistics(original_size, paths_to_patches, paths_to_embedding_dirs, variants)
    save_full_embedding_statistics(cosine_similarities, mean_squared_errors, result_dir)
    save_embedding_statistics(cosine_similarities, mean_squared_errors, result_dir)

    # setup for analysis of the attention maps
    paths_emb_dir_with_attentions = [p for p in paths_to_embedding_dirs if "attentions" in p and "crop" not in p]
    attention_variants = [var for var in variants if "attentions" in var and "crop" not in var]
    if allow_overlap:
        attn_stats_dir = os.path.join(result_dir, f"attn_stats_bw={BASELINE_WIDTH}_overlap")
    else:
        attn_stats_dir = os.path.join(result_dir, f"attn_stats_bw={BASELINE_WIDTH}")

    os.makedirs(attn_stats_dir, exist_ok=True) 

    mse_attns, indices_distances, top_regions_overlap = get_attention_statistics(original_size, paths_to_patches,
                                                                paths_emb_dir_with_attentions, attention_variants, 
                                                                nr_of_compared_max_vals, allow_overlap)
    save_full_attention_statistics(mse_attns, indices_distances, top_regions_overlap, attn_stats_dir, nr_of_compared_max_vals)
    save_attention_statistcs(mse_attns, indices_distances, top_regions_overlap, attn_stats_dir, nr_of_compared_max_vals)


    

if __name__ == '__main__':
    args = parse_arguments()
    main(**args)   
    
	
