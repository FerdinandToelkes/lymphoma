import os
import torch
import pandas as pd
import argparse
import numpy as np
import re 
import matplotlib.pyplot as plt
import matplotlib as mpl

from lymphoma.resnet_ddp.eval_model.plot_saved_results import set_plotting_config




"""
Script implemented for local use, execute in ~/code with:

python3 -m lymphoma.uni_analysis.visualize_results --tex_plot
"""


def parse_arguments() -> dict:
    """ Parse the arguments given to the script. 
    
    Returns:
        dict: Dictionary containing the arguments.
    """
    parser = argparse.ArgumentParser(description='Visualize results of embedding and attention analysis.')
    parser.add_argument('-blw', '--baseline_width', default=400, type=int, help='Width of the baseline taken for computing the side length of the squares to compare top regions of the attentions. This is needed to specify the target directory. (default: 400)')
    parser.add_argument('-os', '--original_size', default=4096, type=int, help='Original size of the patches. (default: 4096)')
    parser.add_argument('-wo', '--with_overlap', default=False, action='store_true', help='Whether overlapping regions were allowed during the comparison of the attentions. (default: False)')
    parser.add_argument('--tex_plot', default=False, action='store_true', help='Whether to save the plots with tex formatting. (default: False)')
    args = parser.parse_args()
    return dict(vars(args))


#########################################################################################################################
####################################### Functions to viszualize embedding results #######################################
#########################################################################################################################


def get_paths_to_embeddings(paths_to_org_embedding: list) -> list:
    """ Get the paths to the embeddings of the patches. 
    
    Args:
        paths_to_org_embedding (list): List of paths to the directories containing the embeddings of the patches.

    Returns:
        list: List of paths to the embeddings of the patches.
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

def get_single_embedding_statistics(original_size: int, paths_to_patches: list, paths_to_embedding_dirs: list, 
                                        variants: list, cosine_similarities: dict, mean_squared_errors: dict) -> tuple[dict]:
    """ Compute the cosine similarity and mean squared error for the embeddings of the different variants. 
    
    Args:
        original_size (int): Original size of the patches.
        paths_to_patches (list): List of paths to the patches.
        paths_to_embedding_dirs (list): List of paths to the directories containing the embeddings of the patches.
        variants (list): List of the variants of the embeddings, e.g. ["4096", "2048", "1024_hflip"].
        cosine_similarities (dict): Dictionary containing the cosine similarities for the different variants.
        mean_squared_errors (dict): Dictionary containing the mean squared errors for the different variants.

    Returns:
        tuple[dict]: Tuple containing the cosine similarities and mean squared errors for the different variants.
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
        cosine_similarities (dict): Dictionary containing the cosine similarities for the different variants.
        mean_squared_errors (dict): Dictionary containing the mean squared errors for the different variants.

    Returns:
        tuple[dict]: Tuple containing the cosine similarities and mean squared errors for the different variants.
    """
    for var in cosine_similarities.keys():
        cosine_similarities[var]["mean"] = np.mean(cosine_similarities[var]["values"])
        cosine_similarities[var]["std"] = np.std(cosine_similarities[var]["values"])
        mean_squared_errors[var]["mean"] = np.mean(mean_squared_errors[var]["values"])
        mean_squared_errors[var]["std"] = np.std(mean_squared_errors[var]["values"])
        cosine_similarities[var]["std_err"] = cosine_similarities[var]["std"] / np.sqrt(len(cosine_similarities[var]["values"]))
        mean_squared_errors[var]["std_err"] = mean_squared_errors[var]["std"] / np.sqrt(len(mean_squared_errors[var]["values"]))
    return cosine_similarities, mean_squared_errors
    


def save_emb_augment_stats_to_latex(df_augment: pd.DataFrame, result_dir: str):
    """ Save the augment statistics to a latex table. 
    
    Args:
        df_augment (pd.DataFrame): Dataframe containing the augment statistics.
        result_dir (str): Directory to save the latex table to.
    """
    path = os.path.join(result_dir, "augment_embedding_statistics.tex")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
    # sort df according to cosine similarity (descending)
    df_augment = df_augment.sort_values(by="Cosine Similarity Avg", ascending=False)
    # make latex table out of above data without the std columns
    df_augment["Cosine Similarity Avg"] = df_augment["Cosine Similarity Avg"].map("{:.4f}".format)
    df_augment["Mean Squared Error Avg"] = df_augment["Mean Squared Error Avg"].map("{:.4f}".format)
    df_augment["Cosine Similarity Std Error"] = df_augment["Cosine Similarity Std Error"].map("{:.4f}".format)
    df_augment["Mean Squared Error Std Error"] = df_augment["Mean Squared Error Std Error"].map("{:.4f}".format)
    del df_augment["Cosine Similarity Std"]
    del df_augment["Mean Squared Error Std"]
    df_augment.to_latex(path, index=False)
    print(f"Saved the augment embedding statistics to {path}")

def plot_embedding_downsample_statistics(result_dir: str, original_size: int, tex_plot: bool):
    """ Plot the cosine similarity and mean squared error for the embeddings at different resolutions. 
    
    Args:
        result_dir (str): Directory to save the plots to.
        original_size (int): Original size of the patches.
        tex_plot (bool): Whether to save the plots with tex formatting.
    """
    compare_name = "4096_attentions_hflip_vflip_color_jitter"
    compare_cos_sim, compare_mse = load_emb_comparison_values(result_dir, compare_name)
    df_crop = load_crop_statistics(original_size, result_dir)
    df_downsample = load_downsample_emb_stats(result_dir)
    x, x_resolutions = extract_x_values_from_df(df_crop)
    set_plotting_config(fontsize=10, aspect_ratio=8/6, width_fraction=0.5, text_usetex=tex_plot)
    plot_attentions_cos_sim(df_crop, df_downsample, x, x_resolutions, result_dir, compare_cos_sim)
    set_plotting_config(fontsize=10, aspect_ratio=8/6, width_fraction=0.5, text_usetex=tex_plot)
    plot_embedding_mse(df_crop, df_downsample, x, x_resolutions, result_dir, compare_mse)

def load_emb_comparison_values(path: str, compare_name: str) -> tuple[float]:
    """ Load the MSE and cosine similarity values of the augmentations. 
    
    Args:
        path (str): Path to the directory containing the embedding statistics.
        compare_name (str): Name of the comparison variant.

    Returns:
        tuple[float]: Tuple containing the cosine similarity and mean squared error of the comparison variant.
    """
    path = os.path.join(path, "embedding_statistics.csv")
    df = pd.read_csv(path)
    df_compared = df[df["Variation"] == compare_name]
    return df_compared["Cosine Similarity Avg"].values[0], df_compared["Mean Squared Error Avg"].values[0]

def load_crop_statistics(original_size: int, path: str) -> pd.DataFrame:
    """ Read in the statistics from the csv file and return only the statistics for the 
        cropping and the original, i.e. the one with highest resolution and no augmentation.

    Args:
        original_size (int): Original size of the patches.
        path (str): Path to the directory containing the embedding statistics.  

    Returns:
        pd.DataFrame: Dataframe containing the cropping statistics.  
    """
    path = os.path.join(path, "embedding_statistics.csv")
    df = pd.read_csv(path)
    # the cropped resolutions are the ones with names {res}_attentions_crop_{res}
    pattern = re.compile(r"^\d+_attentions_crop_\d+$") # ^ start of string, \d digit, + one or more, $ end of string
    df_crop = df[df["Variation"].str.contains(pattern)]
    df_org_res = df[df["Variation"] == f"{original_size}_attentions"]
    return pd.concat([df_org_res, df_crop])

def load_downsample_emb_stats(path: str, name: str = "embedding_statistics.csv") -> pd.DataFrame:
    """ Read in the statistics from the csv file and return only the statistics for the downsampled
        and the original,i.e. the one with highest resolution and no augmentation.
    
    Args:
        path (str): Path to the directory containing the embedding statistics.
        name (str, optional): Name of the embedding statistics file. Defaults to "embedding_statistics.csv".

    Returns:
        pd.DataFrame: Dataframe containing the downsampled embedding
    """
    path = os.path.join(path, name)
    df = pd.read_csv(path)
    # the downsampled resolutions are the ones with names {res}_attentions -> define regular expression to filter them out
    pattern = re.compile(r"^\d+_attentions$") # ^ start of string, \d digit, + one or more, $ end of string
    df_downsample = df[df["Variation"].str.contains(pattern)]
    return df_downsample

def load_emb_augment_statistics(original_size: int, path: str, 
                                name: str = "embedding_statistics.csv") -> pd.DataFrame:
    """ Read in the statistics from the csv file and return only the statistics 
        for the augmented and the original. 
    
    Args:
        original_size (int): Original size of the patches.
        path (str): Path to the directory containing the embedding statistics.
        name (str, optional): Name of the embedding statistics file. Defaults to "embedding_statistics.csv".

    Returns:
        pd.DataFrame: Dataframe containing the embedding statistics.
    """
    path = os.path.join(path, name)
    df = pd.read_csv(path)
    # the augmented resolutions are the ones with names containing hflip, vflip or color_jitter
    pattern = re.compile(r"^(?=.*hflip|.*vflip|.*color_jitter).*$") # ?=. look ahead, * zero or more of the preceding token, | is or
    df_augment = df[df["Variation"].str.contains(pattern)]
    df_org_res = df[df["Variation"] == f"{original_size}_attentions"]
    return pd.concat([df_org_res, df_augment])


def extract_x_values_from_df(df: pd.DataFrame, column_name: str = "Variation") -> tuple[list]:
    """ Extract the x values and the x values as the percentage of the original area. 
    
    Args:
        df (pd.DataFrame): Dataframe containing the embedding statistics.
        column_name (str, optional): Name of the column containing the resolutions. Defaults to "Variation".

    Returns:
        tuple[list]: Tuple containing the x values and the x values as the percentage of the original area.
    """
    x = df[column_name]
    x = [int(x.split("_")[0]) for x in x]
    x.sort(reverse=True)
    x_resolutions = [x**2 for x in x]
    x_resolutions = [x / x_resolutions[0] for x in x_resolutions]
    return x, x_resolutions

def plot_attentions_cos_sim(df_crop: pd.DataFrame, df_downsample: pd.DataFrame, x: list, 
                            path: str, compare_cos_sim: float):
    """ Plot the cosine similarity for the cropped and downsampled images at different resolutions. 
    
    Args:
        df_crop (pd.DataFrame): Dataframe containing the cropping statistics.
        df_downsample (pd.DataFrame): Dataframe containing the downsampled statistics.
        x (list): List of x values.
        path (str): Path to the directory to save the plot to.
        compare_cos_sim (float): Cosine similarity of the comparison variant.
    """
    path = os.path.join(path, "emb_cos_sim_crop_vs_downsampled.pdf")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
    
    y_crop = df_crop["Cosine Similarity Avg"]
    y_crop_std_err = df_crop["Cosine Similarity Std Error"]
    y_down = df_downsample["Cosine Similarity Avg"]
    y_down_std_err = df_downsample["Cosine Similarity Std Error"]
   
    # Create the figure
    fig = plt.figure()
    ax = fig.gca()  # Get current axis

    # Plot cosine similarity for cropped and downsampled images with shaded regions for the standard error
    ax.plot(x, y_crop, 'o-', color='g', label='Cropped')
    ax.plot(x, y_down, 's-', color='b', label='Downsampled')
    ax.fill_between(x, y_down - y_down_std_err, y_down + y_down_std_err, color='b', alpha=0.2)
    ax.fill_between(x, y_crop - y_crop_std_err, y_crop + y_crop_std_err, color='g', alpha=0.2)

    # Add horizontal line for comparison
    ax.axhline(compare_cos_sim, color='r', linestyle='-', linewidth=0.8, label=f'Comparison ({compare_cos_sim:.4f})')

    # Reverse the x-axis and set the ticks and labels for the x-axis
    ax.set_xlim(ax.get_xlim()[::-1])
    # ax.set_xticks(x)
    # ax.set_xticklabels([f'{orig}\n({res:.2f})' for orig, res in zip(x, x_resolutions)])
    # ax.set_xticklabels([f'{res*100:.0f}' for res in x_resolutions])

    # Labels and titles
    ax.set_xlabel('Image Size')
    ax.set_ylabel('Cosine Similarity')

    # Customize gridlines, add a legend and adjust layout to ensure everything fits
    ax.grid(True, which='both', linestyle='--', linewidth=0.6)
    ax.legend()
    fig.tight_layout()

    # Save the figure as a high-resolution image
    plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved the cosine similarity plot to {path}")
    # plt.show()


def plot_embedding_mse(df_crop: pd.DataFrame, df_downsample: pd.DataFrame, x: list, 
                       path: str, compare_mse: float):
    """ Plot the MSE for the cropped and downsampled images at different resolutions. 
    
    Args:
        df_crop (pd.DataFrame): Dataframe containing the cropping statistics.
        df_downsample (pd.DataFrame): Dataframe containing the downsampled statistics.
        x (list): List of x values.
        path (str): Path to the directory to save the plot to.
        compare_mse (float): Mean squared error of the comparison variant
    """
    path = os.path.join(path, "emb_mse_crop_vs_downsampled.pdf")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
    
    y_crop = df_crop["Mean Squared Error Avg"]
    y_crop_std_err = df_crop["Mean Squared Error Std Error"]
    y_down = df_downsample["Mean Squared Error Avg"]
    y_down_std_err = df_downsample["Mean Squared Error Std Error"]
   
    # Create the figure
    fig = plt.figure()
    ax = fig.gca() # Get current axis

    # Plot MSE for cropped and downsampled images with shaded regions for the standard error
    ax.plot(x, y_crop, 'o-', color='g', label='Cropped')
    ax.plot(x, y_down, 's-', color='b', label='Downsampled')
    ax.fill_between(x, y_down - y_down_std_err, y_down + y_down_std_err, color='b', alpha=0.2)
    ax.fill_between(x, y_crop - y_crop_std_err, y_crop + y_crop_std_err, color='g', alpha=0.2)

    # Add horizontal line for comparison
    ax.axhline(compare_mse, color='r', linestyle='-', linewidth=0.8, label=f'Comparison ({compare_mse:.4f})')

    # Reverse the x-axis and set the ticks and labels for the x-axis
    ax.set_xlim(ax.get_xlim()[::-1])
    # ax.set_xticks(x)
    # ax.set_xticklabels([f'{orig}\n({res:.2f})' for orig, res in zip(x, x_resolutions)])
    # ax.set_xticklabels([f'{res*100:.0f}' for res in x_resolutions])

    # Labels and titles
    ax.set_xlabel('Image Size')
    ax.set_ylabel('Mean Squarred Error')

    # Customize gridlines, add a legend and adjust layout to ensure everything fits
    ax.grid(True, which='both', linestyle='--', linewidth=0.6)
    ax.legend()
    fig.tight_layout()

    # Save the figure as a high-resolution image
    plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved the MSE plot to {path}")
    #plt.show()


######################################################################################################################################
################################################## Attention map analysis functions ##################################################
######################################################################################################################################

def save_mse_augment_attn_stats_to_latex(df_augment: pd.DataFrame, result_dir: str):
    """ Save the attention MSE for the augmentations to latex table. 
    
    Args:
        df_augment (pd.DataFrame): Dataframe containing the augment statistics.
        result_dir (str): Directory to save the latex table to.
    """
    path = os.path.join(result_dir, "augment_mse_attn_stats.tex")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
    # sort df according to mse (ascending)
    df_augment = df_augment.sort_values(by="Mean Squared Error Avg", ascending=True)
    # drop the columns which do not contain the mean squared error
    del df_augment["Indices Distance Avg"]
    del df_augment["Indices Distance Std"]
    del df_augment["Indices Distance Std Error"]
    del df_augment["Top Regions Overlap Avg"]
    del df_augment["Top Regions Overlap Std"]
    del df_augment["Top Regions Overlap Std Error"]
    # convert the dataframes to latex tables
    df_augment.to_latex(path, index=False)
    print(f"Saved the augment execution times to {path}")



def save_idx_augment_attn_stats_to_csv(dfs_augment: dict, result_dir: str):
    """ Save the statistics comparing the positions resulting from the attentions 
        for the augmentations to one combined csv file.
    
    Args:
        dfs_augment (dict): Dictionary containing the dataframes for the augmentations.
        result_dir (str): Directory to save the csv file to.
    """
    path = os.path.join(result_dir, "augment_top_idx_attn_stats.csv")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
    
    # Initialize a list to hold the concatenated DataFrames
    combined_df = pd.DataFrame()

    # Loop over each DataFrame (each corresponding to a different number of regions)
    file_names = list(dfs_augment.keys())
    nr_of_regions = [int(f.split("_")[2]) for f in file_names] # name of form attn_stats_{int}_compared_regions.csv
    min_nr = min(nr_of_regions)
    for regs, df in zip(nr_of_regions, dfs_augment.values()):
        # Assuming each DataFrame has a unique number of regions comparison, we'll rename columns accordingly
        # Add a suffix to distinguish columns for each number of regions
        # delete unnecessary columns like mse and all std columns
        df = df.drop(columns=["Mean Squared Error Avg", "Mean Squared Error Std", "Mean Squared Error Std Error",
        "Indices Distance Std", "Top Regions Overlap Std"])
        if regs != min_nr:
            df = df.drop(columns=["Variation"])
        
        suffix = f" {regs} regions"
        df = df.add_suffix(suffix )

        # Combine them horizontally (by columns) if already initialized
        if combined_df.empty:
            combined_df = df.rename(columns={"Variation" + suffix: "Variation"})
        else:
            combined_df = pd.concat([combined_df, df], axis=1)
    # exchange columns with rows since there are too many columns
    # save the names of the columns before exchanging
    file_names = combined_df.columns
    row_name = file_names[0]
    # remove the first column name from file_names
    file_names = file_names[1:]
    combined_df = combined_df.T
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.rename(columns=combined_df.iloc[0]).drop(combined_df.index[0])
    # add the names of the rows
    combined_df.insert(0, row_name, file_names)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(path, index=False)
    print(f"Saved table to {path}")

def save_idx_downscale_attn_stats_to_csv(dfs_downscale: dict, result_dir: str):
    """ Save the statistics comparing the positions resulting from the attentions 
        for the various resolutions to one combined csv file.
        
    Args:
        dfs_downscale (dict): Dictionary containing the dataframes for the downscale resolutions.
        result_dir (str): Directory to save the csv file to.
    """
    path = os.path.join(result_dir, "downscale_top_idx_attn_stats.csv")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
    
    # Initialize a list to hold the concatenated DataFrames
    combined_df = pd.DataFrame()

    # Loop over each DataFrame (each corresponding to a different number of regions)
    file_names = list(dfs_downscale.keys())
    nr_of_regions = [int(f.split("_")[2]) for f in file_names] # name of form attn_stats_{int}_compared_regions.csv
    min_nr = min(nr_of_regions)
    for regs, df in zip(nr_of_regions, dfs_downscale.values()):
        # Assuming each DataFrame has a unique number of regions comparison, we'll rename columns accordingly
        # Add a suffix to distinguish columns for each number of regions
        # delete unnecessary columns like mse and all std columns
        df = df.drop(columns=["Indices Distance Avg","Indices Distance Std", "Indices Distance Std Error", 
                              "Mean Squared Error Std", "Top Regions Overlap Std"])
        if regs != min_nr:
            df = df.drop(columns=["Variation"])
        
        suffix = f" {regs} regions"
        df = df.add_suffix(suffix )

        # Combine them horizontally (by columns) if already initialized
        if combined_df.empty:
            combined_df = df.rename(columns={"Variation" + suffix: "Variation"})
        else:
            combined_df = pd.concat([combined_df, df], axis=1)
  
    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(path, index=False)
    print(f"Saved table to {path}")


def save_idx_augment_attn_stats_to_latex(result_dir: str):
    """ Save the statistics comparing the positions resulting from the attentions 
    for the augmentations to one combined LaTeX table. 
    
    Args:
        result_dir (str): Directory to save the LaTeX table
    """
    overlap_path = os.path.join(result_dir, "augment_top_idx_overlaps.tex")
    distances_path = os.path.join(result_dir, "augment_top_idx_distances.tex")
    if os.path.exists(overlap_path) or os.path.exists(distances_path):
        print(f"Files {overlap_path} or {distances_path} already exists. Will not any of overwrite them.")
        return
    
    # load df and split df according to rows containing overlap and distance
    combined_df = pd.read_csv(os.path.join(result_dir, "augment_top_idx_attn_stats.csv"))
    df_overlap = combined_df[combined_df["Variation"].str.contains("Top Regions Overlap")]
    df_dists = combined_df[combined_df["Variation"].str.contains("Indices Distance")]

    # convert overlap values to percentage
    df_overlap.iloc[:, 1:] = df_overlap.iloc[:, 1:] * 100

    # take for each numerical value only 2 decimal places
    for col in df_overlap.columns[1:]:
        # df_overlap[col] = df_overlap[col].map("{:.2f}".format)
        df_overlap[col] = df_overlap[col].map("{:.2f}".format)
        df_dists[col] = df_dists[col].map("{:.2f}".format)

    # convert the dataframes to latex tables
    df_overlap.to_latex(overlap_path, index=False)
    df_dists.to_latex(distances_path, index=False)
    print(f"Saved LaTeX table to {overlap_path} and {distances_path}")


def get_attn_augment_stats(path: str) -> dict:
    """ Read in the statistics from the csv files computed for different 
    number of reigions and return only the statistics for the augmented dataframes.
    
    Args:
        path (str): Path to the directory containing the attention statistics.

    Returns:
        dict: Dictionary containing the augmented attention statistics.
    """
    # load files, which have pattern attn_stats_{int}_compared_regions.csv
    filename_pattern = re.compile(r"^attn_stats_\d+_compared_regions.csv$") # ^ start of string, \d digit, + one or more, $ end of string
    file_names = [f for f in os.listdir(path) if filename_pattern.match(f)]
    # sort the file names according to the number of regions
    file_names = sorted(file_names, key=lambda x: int(x.split("_")[2]))
    paths = [os.path.join(path, f) for f in file_names]
    dfs = [pd.read_csv(p) for p in paths]
    # The augmented resolutions are the ones with names containing hflip, vflip, or color_jitter
    augment_pattern = r"(?=.*hflip|.*vflip|.*color_jitter)"  # ?=. look ahead, * zero or more of the preceding token, | is or
    dfs_augment = [df[df["Variation"].str.contains(augment_pattern)] for df in dfs]
    return {name: df for name, df in zip(file_names, dfs_augment)} 


def combine_attn_downsample_stats(path: str) -> dict:
    """ Read in the statistics from the csv files computed for different 
        number of reigions and return only the statistics for the overlaps 
        dataframes of different resolutions. 
        
    Args:
        path (str): Path to the directory containing the attention statistics.
        
    Returns:
        dict: Dictionary containing the downsampled attention statistics.
    """
    # load files, which have pattern attn_stats_{int}_compared_regions.csv
    filename_pattern = re.compile(r"^attn_stats_\d+_compared_regions.csv$") # ^ start of string, \d digit, + one or more, $ end of string
    file_names = [f for f in os.listdir(path) if filename_pattern.match(f)]
    # sort the file names according to the number of regions
    file_names = sorted(file_names, key=lambda x: int(x.split("_")[2]))
    paths = [os.path.join(path, f) for f in file_names]
    dfs = [pd.read_csv(p) for p in paths]
    # the downscale resolutions are the ones with names of form {res}_attentions 
    downscale_pattern = r"^\d+_attentions$"  # No parentheses, no match groups
    dfs_down = [df[df["Variation"].str.contains(downscale_pattern)] for df in dfs]
    
    return {name: df for name, df in zip(file_names, dfs_down)} 


def plot_attention_downsample_statistics(dfs_augment: pd.DataFrame, result_dir: str, 
                                         original_size: int, tex_plot: bool):
    """ Plot the mean squared error and the overlap of the top regions for the downsampled 
        attentions at different resolutions and for the case of the overlaps for 
        different number of top regions. 
    
    Args:
        dfs_augment (pd.DataFrame): Dataframe containing the augment statistics.
        result_dir (str): Directory to save the plots to.
        original_size (int): Original size of the patches.
        tex_plot (bool): Whether to save the plots with tex formatting.
    """
    # take augment dataframe corresponding to one region
    df_compare = dfs_augment["attn_stats_1_compared_regions.csv"]
    compare_name = "4096_attentions_hflip_vflip_color_jitter"
    
    # compare value is in column MSE and Overlap Avg and the row with the compare_name
    compare_mse = df_compare[df_compare["Variation"] == compare_name]["Mean Squared Error Avg"].values[0]
    compare_overlap = df_compare[df_compare["Variation"] == compare_name]["Top Regions Overlap Avg"].values[0] * 100
    
    # load the combined statistics for the downsampled attentions and extract the x values
    df_downsample = load_combined_downsample_attn_stats(result_dir, original_size)
    x, x_resolutions = extract_x_values_from_df(df_downsample)
    
    # plot_attention_mse_and_overlaps(df_downsample, x, x_resolutions, result_dir)
    set_plotting_config(fontsize=10, aspect_ratio=8/6, width_fraction=1, text_usetex=tex_plot)
    plot_attn_overlap_mult_regs(df_downsample, x, x_resolutions, result_dir, compare_overlap)
    set_plotting_config(fontsize=10, aspect_ratio=8/6, width_fraction=0.5, text_usetex=tex_plot)
    plot_attention_mse(df_downsample, x, x_resolutions, result_dir, compare_mse)

def load_combined_downsample_attn_stats(path: str, original_size: int, 
                                        name: str = "downscale_top_idx_attn_stats.csv") -> pd.DataFrame:
    """ Read in the attention statistics from the csv file containing the combined statistics 
        for the different resolutions. Add one row for the original resolution which contains a 0 
        for all MSE and a 1 for all overlap columns.
    
    Args:
        path (str): Path to the directory containing the attention statistics.
        original_size (int): Original size of the patches.
        name (str, optional): Name of the attention statistics file. Defaults to "downscale_top_idx_attn_stats.csv".
    
    Returns:
        pd.DataFrame: Dataframe containing the downsampled attention statistics.
    """
    path = os.path.join(path, name)
    df = pd.read_csv(path)
    # add a row for the original resolution
    df_org = create_original_row_from_df(df, original_size)
    df = pd.concat([df_org, df], ignore_index=True)
    return df

def create_original_row_from_df(df: pd.DataFrame, original_size: int) -> pd.DataFrame:
    """ Add a row for the original resolution which contains a 0 for all MSE and a 1 for all overlap columns. 
    
    Args:
        df (pd.DataFrame): Dataframe containing the attention statistics.
        original_size (int): Original size of the patches.

    Returns:
        pd.DataFrame: Dataframe containing the original row.
    """
    # get the names of the columns
    cols = df.columns    
    # create a new row with the original resolution
    new_row = {col: 0 for col in cols}
    new_row["Variation"] = f"{original_size}_attentions"
    for col in cols:
        if "Overlap Avg" in col:
            new_row[col] = 1
    df_org = pd.DataFrame([new_row])
    return df_org

def plot_attn_overlap_mult_regs(df_downsample: pd.DataFrame, x: list, x_resolutions: list, 
                                path: str, compare_overlap: float):
    """ Plot the overlaps for the attentions of downsampled images at different 
        resolutions and for different number of regions that are compared. 
    
    Args:
        df_downsample (pd.DataFrame): Dataframe containing the downsampled statistics.
        x (list): List of x values.
        x_resolutions (list): List of x values as the percentage of the original area.
        path (str): Path to the directory to save the plot to.
        compare_overlap (float): Overlap of the comparison

    Raises:
        ValueError: If there are too many columns to plot.
    """
    path = os.path.join(path, "attn_overlaps_downsampled.pdf")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
 
    # Create the figure
    fig = plt.figure()
    ax = fig.gca()  # Get current axis

    mse_avg_cols = [col for col in df_downsample.columns if "Overlap Avg" in col]
    mse_std_cols = [col for col in df_downsample.columns if "Overlap Std Error" in col]
    colors = plt.cm.viridis(np.linspace(0, 1, len(mse_avg_cols)))  # Get a range of colors
    markers = ['o', 's', 'D', '^', 'v', 'x', '+', '*', 'h', '<']  # 10 different markers
    if len(mse_avg_cols) > len(markers):
        raise ValueError(f"Too many columns to plot. Only {len(markers)} markers available.")
    else:
        markers = markers[:len(mse_avg_cols)]

    for avg_col, std_col, color, marker in zip(mse_avg_cols, mse_std_cols, colors, markers):
        # Extract the data for the current resolution
        y = df_downsample[avg_col] * 100  # Convert to percentage
        y_std_err = df_downsample[std_col] * 100  # Convert to percentage
        # Plot MSE for cropped and downsampled images with shaded regions for the standard error
        label = avg_col.split(' ')[4] # Adjust to capture the unique part of the column name
        label = f"{label} regions" if label != "1" else "1 region"
        ax.plot(x, y, marker=marker, color=color, label=label)
        ax.fill_between(x, y - y_std_err, y + y_std_err, color=color, alpha=0.2)

    # Reverse the x-axis and set the ticks and labels for the x-axis
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xticks(x)
    ax.set_xticklabels([f'{orig}\n({res*100:.2f})' for orig, res in zip(x, x_resolutions)])

    # Add horizontal line for comparison
    hline = ax.axhline(compare_overlap, color='r', linestyle='-', linewidth=0.8, label=f'Comparison ({compare_overlap:.2f})')

    # Labels and titles
    ax.set_xlabel('Image Size (Proportion of Original Area in $\%$)')
    ax.set_ylabel('Overlap (in $\%$)')

    # Colormap and normalization for the color bar
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=1, vmax=len(mse_avg_cols))  # Adjust range to match the number of regions
    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy array for the color bar
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'Number of Regions', rotation=270, labelpad=10)
    cbar.set_ticks(range(1, len(mse_avg_cols) + 1))
    cbar.set_ticklabels([f"{i}" for i in range(1, len(mse_avg_cols) + 1)])

    # Customize gridlines, add a legend and adjust layout to ensure everything fits
    ax.grid(True, which='both', linestyle='--', linewidth=0.6)
    # only show legend for the horizontal line
    ax.legend(handles=[hline], loc="lower left")
    fig.tight_layout()

    # Save the figure as a high-resolution image
    plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved the Overlap plot to {path}")



def plot_attention_mse(df_downsample: pd.DataFrame, x: list, x_resolutions: list, path: str, compare_mse: float):
    """ Plot the MSE for the attentions of downsampled images at different resolutions. Note that the 
        MSE is independent of the number of regions that are compared.
    
    Args:
        df_downsample (pd.DataFrame): Dataframe containing the downsampled statistics.
        x (list): List of x values.
        x_resolutions (list): List of x values as the percentage of the original area.
        path (str): Path to the directory to save the plot to.
        compare_mse (float): Mean squared error of the comparison
    """
    path = os.path.join(path, "attn_mse_downsampled.pdf")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
 
    # get two arbitrary atching columns to extract the MSE and the standard error
    cols_mse = df_downsample.columns
    col_mse = [col for col in cols_mse if "Mean Squared Error Avg" in col]
    col_mse = col_mse[0]
    col_std_err = col_mse.replace("Avg", "Std Error")

    # take first two columns, since MSE does not depend on the number of regions
    y = df_downsample[col_mse]
    y_std_err = df_downsample[col_std_err]
   
    # Create the figure
    fig = plt.figure()
    ax = fig.gca()  # Get current axis

    # Plot MSE for cropped and downsampled images with shaded regions for the standard error
    ax.plot(x, y, 's-', color='b', label='MSE')
    ax.fill_between(x, y - y_std_err, y + y_std_err, color='b', alpha=0.2)

    # Add horizontal line for comparison
    ax.axhline(compare_mse, color='r', linestyle='-', linewidth=0.8, label=f'Comparison ({compare_mse:.2e})')

    # Reverse the x-axis and set the ticks and labels for the x-axis
    ax.set_xlim(ax.get_xlim()[::-1])
    # ax.set_xticks(x)
    # ax.set_xticklabels([f'{orig}\n({res:.2f})' for orig, res in zip(x, x_resolutions)])
    # ax.set_xticklabels([f'{res*100:.0f}' for res in x_resolutions])

    # Labels and titles
    ax.set_xlabel('Image Size') 
    ax.set_ylabel('Mean Squarred Error')

    # Customize gridlines, add a legend and adjust layout to ensure everything fits
    ax.grid(True, which='both', linestyle='--', linewidth=0.6)
    ax.legend(loc= "upper left")
    fig.tight_layout()

    # Save the figure as a high-resolution image
    plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved the MSE plot to {path}")


#####################################################################################################################################
################################################# Execution time analysis functions #################################################
#####################################################################################################################################

def load_timings(result_dir: str) -> dict:
    """ Load the execution times for the embeddings and process them into a nice dictionary. 
    
    Args:
        result_dir (str): Directory containing the timings.txt file.

    Returns:
        dict: Dictionary containing the execution times
    """
    # path_to_timings = os.path.join("/data", f"{patch_size}_patches", target_dir, "timings.txt")
    path_to_timings = os.path.join(result_dir, "timings.txt")
    with open(path_to_timings, "r") as f:
        timings = f.readlines()
    # split the lines into variant and time
    timings = [t.strip().split(":") for t in timings] 
    timings = timings[1:] # remove the header
    timings = [t for t in timings if len(t) == 2] # e.g no empty lines
    timings = {t[0]: float(t[1]) for t in timings}
    return timings

def split_timings(timings: dict, original_size: int) -> tuple[dict]:
    """ Split the timings into downsample, crop and augment timings, always including the timing 
        of the full resolution for comparison. 

    Args:
        timings (dict): Dictionary containing the execution times.
        original_size (int): Original size of the patches.

    Returns:
        tuple[dict]: Tuple containing the downsample, crop and augment timings.    
    """
    # split the timings into downsample, crop and augment timings
    downsample_pattern = re.compile(r"^\d+_attentions$")
    crop_pattern = re.compile(r".*_crop_.*")
    augment_pattern = re.compile(r"^(?=.*hflip|.*vflip|.*color_jitter).*$")

    # 'original size' timings first
    cropping_timings = {f"{original_size}_attentions": timings[f"{original_size}_attentions"]}
    augment_timings = {f"{original_size}_attentions": timings[f"{original_size}_attentions"]}

    # Update these dictionaries with the filtered timings
    cropping_timings.update({key: value for key, value in timings.items() if crop_pattern.match(key)})
    augment_timings.update({key: value for key, value in timings.items() if augment_pattern.match(key)})
    # Downsample timings can be created directly as it doesn't need the 'original size' at the beginning
    downsample_timings = {key: value for key, value in timings.items() if downsample_pattern.match(key)}
    return downsample_timings, cropping_timings, augment_timings

def plot_execution_timings(df_crop: pd.DataFrame, df_downsample: pd.DataFrame, x: list, 
                             x_resolutions: list, path: str):
    """ Plot the execution timings for the cropped and downsampled images at different resolutions.
    
    Args:
        df_crop (pd.DataFrame): Dataframe containing the cropping statistics.
        df_downsample (pd.DataFrame): Dataframe containing the downsampled statistics.
        x (list): List of x values.
        x_resolutions (list): List of x values as the percentage of the original area.
        path (str): Path to the directory to save the plot to.
    """
    path = os.path.join(path, "times_cropped_vs_downsampled.pdf")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
    
    # get y values
    y_crop = df_crop["Execution Time (seconds)"]
    y_down = df_downsample["Execution Time (seconds)"]
   
    # Create the figure
    fig = plt.figure()
    ax = fig.gca()  # Get current axis

    # Plot timings for cropped and downsampled images with log scale
    ax.plot(x, y_crop, 'o-', color='g', label='Cropped')
    ax.plot(x, y_down, 's-', color='b', label='Downsampled')

    # Set the y-axis to log scale
    ax.set_yscale('log')

    # Reverse the x-axis and set the ticks and labels for the x-axis
    ax.set_xlim(ax.get_xlim()[::-1])
    # ax.set_xticks(x)
    # ax.set_xticklabels([f'{orig}\n({res:.2f})' for orig, res in zip(x, x_resolutions)])
    # ax.set_xticklabels([f'{res*100:.0f}' for res in x_resolutions])

    # Labels and titles
    ax.set_xlabel('Image Size')
    ax.set_ylabel('Time in Seconds')

    # Customize gridlines, add a legend and adjust layout to ensure everything fits
    #ax.grid(True, which='both', linestyle='--', linewidth=0.6)
    ax.legend()
    fig.tight_layout()

    # Save the figure as a high-resolution image
    plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved the timing plot to {path}")

def save_augment_time_stats_to_latex(df_augment: pd.DataFrame, result_dir: str):
    """ Save the execution times for the augmentations to a latex table. 
    
    Args:
        df_augment (pd.DataFrame): Dataframe containing the augment statistics.
        result_dir (str): Directory to save the latex table to.
    """
    path = os.path.join(result_dir, "augment_times_statistics.tex")
    if os.path.exists(path):
        print(f"File {path} already exists. Will not overwrite it.")
        return
    # make latex table out of above data without the std columns
    df_augment["Execution Time (seconds)"] = df_augment["Execution Time (seconds)"].map("{:.2f}".format)
    df_augment.to_latex(path, index=False)
    print(f"Saved the augment execution times to {path}")


###############################################################################################################################


def main(baseline_width: int, original_size: int, with_overlap: bool, tex_plot: bool):
    """ Main function to visualize the results of the analysis.

    Args:
        baseline_width (int): Baseline width of the patches.
        original_size (int): Original size of the patches.
        with_overlap (bool): Whether to include overlap in the analysis.
        tex_plot (bool): Whether to save the plots with tex formatting.
    """
    # setup for the visalization of analysis results
    print("Visualizing results")
    #paths_to_patches, paths_to_embedding_dirs, variants, result_dir = setup_embedding_analysis(target_dir, patch_size)
    result_dir = "/Users/ferdinandtolkes/code/lymphoma/uni_analysis/results"
    os.makedirs(result_dir, exist_ok=True)

    # Load statistics for the augmentations and save them to latex
    df_augment = load_emb_augment_statistics(original_size, result_dir, name="embedding_statistics.csv")
    save_emb_augment_stats_to_latex(df_augment, result_dir)

    # Plot the cosine similarity and MSE for the cropped and downsampled images at different resolutions
    plot_embedding_downsample_statistics(result_dir, original_size, tex_plot)



    # setup for plotting results of the attention map analysis
    attn_stats_dir = os.path.join(result_dir, f"attn_stats_bw={baseline_width}")
    if with_overlap:
        attn_stats_dir += "_overlap"
    os.makedirs(attn_stats_dir, exist_ok=True) 

    # load augment statistics, combine and save them to latex and csv
    print(f"loading from {attn_stats_dir}")
    dfs_augment = get_attn_augment_stats(attn_stats_dir)
    first_key = next(iter(dfs_augment))
    save_mse_augment_attn_stats_to_latex(dfs_augment[first_key], attn_stats_dir)
    save_idx_augment_attn_stats_to_csv(dfs_augment, attn_stats_dir)
    save_idx_augment_attn_stats_to_latex(attn_stats_dir)

    # load overlap statistics and save them to latex
    dfs_down = combine_attn_downsample_stats(attn_stats_dir)
    save_idx_downscale_attn_stats_to_csv(dfs_down, attn_stats_dir)

    # plot the attention statistics for the downsampled images
    plot_attention_downsample_statistics(dfs_augment, attn_stats_dir, original_size, tex_plot)
    

    # read in timing .txt file and transform it into a dictionary
    timings = load_timings(result_dir)

    # split the timings into downsample, crop and augment timings
    downsample_timings, cropping_timings, augment_timings = split_timings(timings, original_size)

    # transform the timings into dataframe for consistency
    df_augment = pd.DataFrame(augment_timings.items(), columns=["Variant", "Execution Time (seconds)"])
    save_augment_time_stats_to_latex(df_augment, result_dir)

    # plot the timings for the cropped and downsampled images at different resolutions
    df_downsample = pd.DataFrame(downsample_timings.items(), columns=["Variant", "Execution Time (seconds)"])
    df_crop = pd.DataFrame(cropping_timings.items(), columns=["Variant", "Execution Time (seconds)"])
    x, x_resolutions = extract_x_values_from_df(df_crop, column_name="Variant")
    set_plotting_config(fontsize=10, aspect_ratio=8/6, width_fraction=0.5, text_usetex=tex_plot)
    plot_execution_timings(df_crop, df_downsample, x, x_resolutions, result_dir)


    

if __name__ == '__main__':
    args = parse_arguments()
    main(**args)   
    
	
