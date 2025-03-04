import os
import time
import argparse
import re
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as T 
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

from torch.utils.data import DataLoader
from tqdm import tqdm

from matplotlib.lines import Line2D


from lymphoma.diagnosis_maps import LABELS_MAP_INT_TO_STRING, LABELS_MAP_STRING_TO_INT
from lymphoma.resnet_ddp.eval_model.plot_saved_results import set_plotting_config
from lymphoma.resnet_ddp.eval_model.slide_tester import PatchesTestDataset

# docker command to run this script for models based on patches
"""
screen -dmS plot_slide_attentions sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name ftoelkes_run01 -it -u `id -u $USER` --rm -v /home/ftoelkes/code:/mnt -v /home/ftoelkes/preprocessed_data/test_data:/data ftoelkes_lymphoma python3 -m lymphoma.resnet_ddp.eval_model.plot_slide_attentions --patch_size="1024um" --dataset="kiel" --data_specifier="patches"; exec bash'
"""

# or locally from code
"""
python3 -m lymphoma.resnet_ddp.eval_model.plot_overlay --patch_size="1024um" --dataset="kiel" --data_specifier="patches" --slide_name="94933-2025-0-HE-DLBCL"
"""


def parse_arguments() -> dict:
    """ Parses the arguments for the script. 
    
    Returns:
        dict: Dictionary with the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the (augemented) test set')
    # test parameters
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply", "munich" or "all_data". (default: kiel)')
    parser.add_argument('--slide_name', default='DLBCL/94933-2025-0-HE-DLBCL', type=str, help='Name of the slide to evaluate. (default: DLBCL/94933-2025-0-HE-DLBCL)')
    parser.add_argument('--data_specifier', default='patches', type=str, help='String specifying the patches to be used, e.g top_5_patches (default: "patches")')
    args = parser.parse_args()
    return dict(vars(args))

 
def prepare_patches_list(path_to_slide: str) -> list:
    """ Prepares the list of patches for the dataloader. 
    
    Args:
        path_to_slide (str): Path to the slide containing the patches.

    Returns:
        list: List of patch names.
    """
    patch_names = os.listdir(path_to_slide)
    patch_names = [patch_name for patch_name in patch_names if patch_name.startswith("patch") and patch_name.endswith(".pt")]
    if len(patch_names) == 0:
        raise ValueError(f"No attentions found for slide {path_to_slide}")
    # happens anyway in the dataloader and here just for consistency
    patch_names = sorted(patch_names, key=lambda x: int(x.split("_")[1]))
    return patch_names

def retrieve_patch_size_from_log(path_to_log: str) -> int:
    """ Retrieves the patch size in pixels from the log file of the given slide. 
    
    Args:
        path_to_log (str): Path to the log file of the slide.

    Returns:
        int: Patch size in pixels.
    """
    # get the original patch size in pixels from log file, before resizing happend
    with open(path_to_log, "r") as f:
        lines = f.readlines()
        # search for lines containing Patch size in micro meters and Pixels per m of the WSI
        for line in lines:
            if "Patch size in micro meter" in line:
                patch_size_um = int(line.split(":")[-1])
            if "Pixels per m of the WSI" in line:
                pixels_per_m = int(line.split(":")[-1])
    if not patch_size_um or not pixels_per_m:
        raise ValueError("Patch size in micro meters or Pixels per m of the WSI not found in log file.")
    # calculate the original patch size in pixels
    patch_size_in_pixels = int(patch_size_um * pixels_per_m / 1e6)
    return patch_size_in_pixels


def get_coordinates_from_tiles_list(patch_names: list[str], tile_size_in_pixels: int = 512) -> tuple:
    """ Retrieves the coordinates of the patches of a slide needed for plotting from list of patch names. 
        The coordinates are transformed to absolute coordinates.
    
    Args:
        patch_names (list): List of patch names.
        tile_size_in_pixels (int, optional): Size of the tiles in pixels. (default: 512)
    """
    # retrieve the coordinates from the patch names with regular expressions
    tile_coords = [get_tile_coords_from_filename(patch) for patch in patch_names]

    # compute the absolute coordinates, i.e, combine tile coordinates and patch coordinates 
    abs_coords = [(coords[0]*tile_size_in_pixels, coords[1]*tile_size_in_pixels) for coords in tile_coords]
    return abs_coords

def get_limits_of_plot(coordinates: list, patch_size_in_pixels: int) -> tuple:
    """ Calculates the limits of the plot. 
    
    Args:
        coordinates (list): List of coordinates of the patches.
        patch_size_in_pixels (int): Size of the patches in pixels.

    Returns:
        tuple: Tuple containing the limits of the plot.
    """
    # get x and y coordinates of all patches
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]
    # get min and max values
    buffer = patch_size_in_pixels//2
    x_min = min(x_coords) - buffer
    x_max = max(x_coords) + patch_size_in_pixels + buffer
    y_min = min(y_coords) - patch_size_in_pixels - buffer # since the origin is in the top left corner after inverting the y-axis
    y_max = max(y_coords) + buffer
    return x_min, x_max, y_min, y_max


def get_tile_coords_from_filename(patch_name: str) -> tuple:
    """ Extract tile coordinates: 'patch_0_coords_(35,200,7).pt' -> (35,200) 
    
    Args:
        patch_name (str): Name of the patch.

    Returns:
        tuple: Tuple containing the x and y tile coordinates.
    """
    # Define the regular expression pattern to extract coordinates within round parentheses
    # note: - backslashes are used to escape special characters in the pattern
    #       - parentheses are used to capture the digits within them -> group(1) and group(2)
    tile_coords_pattern = r"\((\d+),(\d+),\d+\)"  

    # Search for the pattern in the filename
    tile_match = re.search(tile_coords_pattern, patch_name)

    # If a match is found, extract the coordinates
    if tile_match:
        x_tile = int(tile_match.group(1))
        y_tile = int(tile_match.group(2))
    else:
        raise ValueError(f"No tile and/or multiplier found in filename {patch_name}.")
    
    return x_tile, y_tile


def plot_attention_at_coord(ax: plt.Axes, img: torch.Tensor, coordinates: list, 
                            patch_size_in_pixels: int, i: int) -> None:
    """ Plots the patch and a colored square at the coordinates on top of the patch according to the prediction. 
    
    Args:
        ax (plt.Axes): The axes to plot on.
        img (torch.Tensor): The image patch.
        coordinates (list): List of coordinates of the patches.
        patch_size_in_pixels (int): Size of the patches in pixels.
        i (int): Index of the patch in the dataloader.
    """
    # Standardization -> ensure with vmin and vmax that all values are within the range of the colormap
    img = (img - img.mean()) / img.std()  

    # plot patch and a colored square at coord on top of the patch according to the prediction
    coord = coordinates[i]
    extent = [coord[0], coord[0]+patch_size_in_pixels, coord[1], coord[1]-patch_size_in_pixels]
    im = ax.imshow(img, cmap="coolwarm",vmin=-2, vmax=2, extent=extent, alpha=0.8)
    return im




def process_patch(img: torch.Tensor) -> torch.Tensor:
    """ Processes the patch: resizing and reordering the channels.
    
    Args:
        img (torch.Tensor): The image patch.

    Returns:
        torch.Tensor: The processed image patch.
    """
    # Resize and permute so that the image is HxWxC, dpi=300 limits the resolution anyways
    processed_img = T.Resize((1024, 1024), antialias=True)(img.unsqueeze(0)).squeeze(0)
    processed_img = processed_img.permute(1, 2, 0)
    return processed_img

def plot_patch_on_axes(axes: list, processed_img: torch.Tensor, coord: list, patch_size_in_pixels: int) -> None:
    """ Plots the same processed image on multiple axes with the specified extent.
    
    Args:
        axes (list): List of axes to plot on.
        processed_img (torch.Tensor): The processed image patch.
        coord (list): Coordinates of the patch.
        patch_size_in_pixels (int): Size of the patches in pixels.
    """
    extent = [coord[0], coord[0] + patch_size_in_pixels,
              coord[1], coord[1] - patch_size_in_pixels]
    for ax in axes:
        ax.imshow(processed_img, extent=extent)



def setup_paths(path_to_data_dir: str, slide_name: str, data_dir: str) -> tuple:
    """ Sets up the paths to the slide, the patch names and the log file.

    Args:
        path_to_data_dir (str): Path to the data directory.
        slide_name (str): Name of the slide.
        data_dir (str): Name of the data directory.

    Returns:
        tuple: Tuple containing the path to the slide, the patch names and the path to the log file.
    """
    path_to_slide = os.path.join(path_to_data_dir, slide_name)
    print(f"Path to slide: {path_to_slide}")
    if not os.path.exists(path_to_slide):
        raise ValueError(f"Path {path_to_slide} does not exist.")
    patch_names = prepare_patches_list(path_to_slide)
    
    path_to_log = path_to_slide.replace(data_dir, "data_dir") + "/sampling.log"
    print(f"Path to log: {path_to_log}")
    if not os.path.exists(path_to_log):
        raise ValueError(f"Path {path_to_log} does not exist.")
    return path_to_slide, patch_names, path_to_log

def get_slide_dataloader(path_to_slide: str, patch_names: list) -> DataLoader:
    """ Returns the dataloader for the slide.

    Args:
        path_to_slide (str): Path to the slide.
        patch_names (list): List of patch names.

    Returns:
        DataLoader: Dataloader for the slide.
    """
    one_slide_dataset = PatchesTestDataset(patch_names, path_to_slide)
    one_slide_dataloader = DataLoader(one_slide_dataset, batch_size=1, shuffle=False)
    return one_slide_dataloader

def setup_subplot(ax: plt.Axes, x_min: int, x_max: int, y_min: int, y_max: int, title: str) -> None:
    """ Setup the subplot with the limits of the plot, title and remove ticks. 
    
    Args:
        ax (plt.Axes): The axes to plot on.
        x_min (int): Minimum x value.
        x_max (int): Maximum x value.
        y_min (int): Minimum y value.
        y_max (int): Maximum y value.
        title (str): Title of the plot.
    """
    # set limits of the plot
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max) 

    # invert y-axis to match the image coordinates -> origin in the top left corner
    ax.invert_yaxis()

    # remove ticks and set title
    ax.set_xticks([])
    ax.set_yticks([])
    fontsize = plt.rcParams["font.size"]
    ax.set_title(title, fontsize=fontsize)

def plot_patches(ax: plt.Axes, one_slide_dataloader: DataLoader, coordinates: list, 
                 patch_size_in_pixels: int) -> None:
    """ Plots the patches of a slide. 
    
    Args:
        ax (plt.Axes): The axes to plot on.
        one_slide_dataloader (DataLoader): Dataloader for the slide.
        coordinates (list): List of coordinates of the patches.
        patch_size_in_pixels (int): Size of the patches in pixels.
    """
    for i, (x, y) in tqdm(enumerate(one_slide_dataloader), total=len(one_slide_dataloader)):
        img = x[0] # first patch of batch with size 1
        im = plot_attention_at_coord(ax, img, coordinates, patch_size_in_pixels, i)
    return im

def plot_patches_multiple_axes(axes: plt.Axes, one_slide_dataloader: DataLoader, 
                               coordinates: list, patch_size_in_pixels: int) -> None:
    """ Plots the patches of a slide. 
    
    Args:
        axes (list): List of axes to plot on.
        one_slide_dataloader (DataLoader): Dataloader for the slide.
        coordinates (list): List of coordinates of the patches.
        patch_size_in_pixels (int): Size of the patches in pixels.
    """
    for i, (x, y) in tqdm(enumerate(one_slide_dataloader), total=len(one_slide_dataloader)):
        img = x[0] # first patch of batch with size 1
        processed_img = process_patch(img)
        # Get the corresponding coordinates
        coord = coordinates[i]
        im = plot_patch_on_axes(axes, processed_img, coord, patch_size_in_pixels)
        # if i > 10:
        #     break
    return im

def get_colors(slide_name: str) -> list:
    """ Returns the colors for the overlay plot with green for ground truth. 
    
    Args:
        slide_name (str): Name of the slide.

    Returns:
        list: List of colors.
    """
    colors = ['white', 'blue', 'orange', 'purple', 'yellow', 'red']
    ground_truth_label = slide_name.split("-")[-1]
    ground_truth_int = LABELS_MAP_STRING_TO_INT[ground_truth_label]
    colors.insert(ground_truth_int, 'green')
    return colors

def overlay_predictions(ax: plt.Axes, preds: dict, coordinates: list, 
                        patch_size_in_pixels: int, colors: list) -> None:
    """ Overlays the predictions on the image patches. 
    
    Args:
        ax (plt.Axes): The axes to plot on.
        preds (dict): Dictionary containing the predictions.
        coordinates (list): List of coordinates of the patches.
        patch_size_in_pixels (int): Size of the patches in pixels.
        colors (list): List of colors for the overlay plot.
    """
    for coord in coordinates:
        color_idx = preds[coord][0]
        rectangle = patches.Rectangle((coord[0], coord[1]-patch_size_in_pixels), patch_size_in_pixels, patch_size_in_pixels, edgecolor='none', facecolor=colors[color_idx], alpha=0.3)
        ax.add_patch(rectangle)

def add_legend(ax: plt.Axes, colors: list) -> None:
    """ Adds a legend to the predictions plot. 
    
    Args:
        ax (plt.Axes): The axes to plot on.
        colors (list): List of colors for the overlay plot.
    """
    labels = [LABELS_MAP_INT_TO_STRING[i] for i in range(7)]
    labels.remove("Unknown")
    colors = [colors[LABELS_MAP_STRING_TO_INT[label]] for label in labels]

    handles = [
        Line2D([0], [0],
            marker='o', 
            linestyle='None',  # No connecting line
            markerfacecolor=color,
            markeredgecolor='none', 
            markersize=2,     # Smaller circle
            label=label)
        for color, label in zip(colors, labels) #skip unknown class
    ]

    ax.legend(
        handles=handles,
        loc='lower left',   # inside the axis at lower left
        fontsize=4,         # smaller text
        borderaxespad=0.1,  # space between legend and axis
        labelspacing=0.4,   # reduce vertical space between entries
        handletextpad=0.5,  # space between marker and text
        handlelength=1.0,   # length of the "marker" area
        frameon=False       # no border
    )

def add_colorbar(fig: plt.Figure, gs: gridspec.GridSpec, ax: plt.Axes, im: plt.Axes) -> None:
    """ Adds a colorbar to the plot. 
    
    Args:
        fig (plt.Figure): The figure to plot on.
        gs (gridspec.GridSpec): The grid specification.
        ax (plt.Axes): The axes to plot on.
        im (plt.Axes): The image handle.
    """
    # Match the colorbar height to the third subplot
    cax = fig.add_subplot(gs[3])  # Extra column for colorbar
    pos = ax.get_position()  # Get the position of the last plot
    cax.set_position([pos.x1 + 0.01, pos.y0, 0.02, pos.height])  # Align it

    if im is None:
        raise ValueError("plotting function must return an image handle from ax.imshow()")
    cbar = plt.colorbar(im, cax=cax)  
    # Set custom tick labels
    cbar.set_ticks([-1.8, 1.8])  # Positions at the bottom and top of colormap
    cbar.set_ticklabels(["Low", "High"], fontsize=8)  # Replace numbers with text
    cbar.ax.tick_params(size=0)  # Hide tick marks for a cleaner look

    # Remove black borders around the colorbar
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

def save_plot(slide_name: str) -> None:
    """ Saves the plot to the output directory. 
    
    Args:
        slide_name (str): Name of the slide.
    """
    output_dir = f"/Users/ferdinandtolkes/code/lymphoma/figures/{slide_name}"
    os.makedirs(output_dir, exist_ok=True)    
    path = os.path.join(output_dir, "wsi_overlay_attention.pdf")
    print(f"Saving plot to: {path}")
    plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def main(patch_size: str, dataset: str, slide_name: str, data_specifier: str):
    """ Main function to plot the overlay of the predictions and the attentions.

    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset.
        slide_name (str): Name of the slide.
        data_specifier (str): String specifying the patches to be used.
    """
    # path_to_data_dir = f"/data/{patch_size}/{dataset}/{data_specifier}/data_dir"
    path_to_data_dir = f"/Users/ferdinandtolkes/code/lymphoma/figures/data_dir"
    print(f"Path to image patches: {path_to_data_dir}")

    path_to_slide, patch_names, path_to_log = setup_paths(path_to_data_dir, slide_name, "data_dir")
    
    # get infos important for plotting
    patch_size_in_pixels = retrieve_patch_size_from_log(path_to_log)
    coordinates = get_coordinates_from_tiles_list(patch_names, tile_size_in_pixels=512)
    x_min, x_max, y_min, y_max = get_limits_of_plot(coordinates, patch_size_in_pixels)

    # get the dataloader for the slide
    image_dataloader = get_slide_dataloader(path_to_slide, patch_names)

    # create empty plot on which we place the patches where they belong using the coordinates
    set_plotting_config(fontsize=10, aspect_ratio=3/1.4, width_fraction=1.0, text_usetex=True)
    figsize = plt.rcParams["figure.figsize"]
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    
    setup_subplot(axs[0], x_min, x_max, y_min, y_max, "WSI")
    setup_subplot(axs[1], x_min, x_max, y_min, y_max, "Predictions")
    setup_subplot(axs[2], x_min, x_max, y_min, y_max, "Attention")

    plot_patches_multiple_axes(axs, image_dataloader, coordinates, patch_size_in_pixels)
        
    ###################### PLOT OVERLAY ######################
    print(f"Plotting overlay...")
    path_to_preds = f"/Users/ferdinandtolkes/code/lymphoma/resnet_ddp/experiments/1024um/kiel/models_on_patches/inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0/RESNET18,dd=embeddings_dir,e=50,lr=0.001,bs=256,wl=True,ls=0.1,p=10,s=42/kiel/94933-2025-0-HE-DLBCL/predictions_for_overlay_plot.npy"
    preds = np.load(path_to_preds, allow_pickle=True).item()

    colors = get_colors(slide_name)
    overlay_predictions(axs[1], preds, coordinates, patch_size_in_pixels, colors)
    add_legend(axs[1], colors)
    

    ###################### PLOT ATTENTIONS ######################
    path_to_attentions_dir = f"/Users/ferdinandtolkes/code/lymphoma/figures/attentions_dir"
    path_to_slide, patch_names, path_to_log = setup_paths(path_to_attentions_dir, slide_name, "attentions_dir")
    print(f"Path to attentions: {path_to_attentions_dir}")

    # get the dataloader and plot the attentions
    attention_dataloader = get_slide_dataloader(path_to_slide, patch_names)
    print(f"Plotting attentions...")
    im = plot_patches(axs[2], attention_dataloader, coordinates, patch_size_in_pixels)
    add_colorbar(fig, gs, axs[2], im)
    
    
    # save fig to output directory
    save_plot(slide_name)
    


if __name__ == "__main__":
    args = parse_arguments()
    start = time.time()
    main(**args)
    end = time.time()
    print(f"Testing took {end-start:.2f}s")


