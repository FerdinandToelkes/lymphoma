import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import sqlite3
import torch
import torchvision.transforms as T

from PIL import Image
from io import BytesIO

from lymphoma.resnet_ddp.eval_model.plot_saved_results import set_plotting_config


# script for visualizing the results of the model evaluation on the test data for local usage because its easier to modify the plots
# execute from code directory with:
"""
python3 -m lymphoma.figures.visualize_preprocessing --tex_plot
"""



def parse_arguments() -> dict:
    """ Parse command-line arguments.

    Returns:
        dict: Dictionary containing the command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Visualize preprocessing with Pamly and Patchcraft.')
    parser.add_argument('--tex_plot', default=False, action='store_true', help='Whether to use LaTeX for text rendering. (default: False)')
    args = parser.parse_args()
    return dict(vars(args))

def get_wsi(path_to_slide: str, resolution_level: int=6):
    """ Get the whole slide image (WSI) at a given resolution level from a SQLite database.

    Args:
        path_to_slide (str): Path to the SQLite database containing the WSI.
        resolution_level (int): Resolution level of the WSI to extract. (default: 6)
    
    Returns:
        np.ndarray: The WSI as a NumPy array.
    """
    wsi = []
    nr_tiles = 2**resolution_level
    for y in range(nr_tiles):
        x_tiles = []
        for x in range(nr_tiles):
            x_tiles.append(get_tile(path_to_slide, x, y, resolution_level))
        wsi.append(np.concatenate(x_tiles, axis=1))
    wsi = np.concatenate(wsi, axis=0)
    return wsi

def get_patch(highest_level: int, level: int, x_tile: int, y_tile: int, path_to_slide: str):
    """ Get a patch from a whole slide image (WSI) at a given resolution level.

    Args:
        highest_level (int): Highest resolution level of the WSI.
        level (int): Resolution level of the patch.
        x_tile (int): x-coordinate of the patch.
        y_tile (int): y-coordinate of the patch.
        path_to_slide (str): Path to the SQLite database containing the WSI.

    Returns:
        np.ndarray: The patch as a NumPy array.
    """
    multiplier = 2**(highest_level-level)
    
    x_coords = [x_tile*multiplier + i for i in range(multiplier)]
    y_coords = [y_tile*multiplier + i for i in range(multiplier)]

    patch = []
    for y in y_coords:
        x_tiles = []
        for x in x_coords:
            x_tiles.append(get_tile(path_to_slide, x, y, highest_level))
        patch.append(np.concatenate(x_tiles, axis=1))
    patch = np.concatenate(patch, axis=0)
    return patch

def get_tile(path_to_slide: str, x: int, y: int, level: int) -> np.ndarray:
    """ Get a tile from a whole slide image (WSI) at a given resolution level.

    Args:
        path_to_slide (str): Path to the SQLite database containing the WSI.
        x (int): x-coordinate of the tile.
        y (int): y-coordinate of the tile.
        level (int): Resolution level of the tile.

    Returns:
        np.ndarray: The tile as a NumPy array.
    """
    with sqlite3.connect(path_to_slide) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT jpeg FROM tiles WHERE x=? AND y=? AND level=? LIMIT 1", (x, y, level))
        row = cursor.fetchone()
        if row is not None:
            # Extract the image data from the row
            jpeg_data = row[0]
            # Create a BytesIO object from the image data
            with BytesIO(jpeg_data) as bytes_io:
                # Read the image from the BytesIO object using PIL.Image.open()
                with Image.open(bytes_io) as pil_image: # returns an image array in NumPy format
                    # Convert the PIL image to a NumPy array and append to y_tiles
                    patch = np.array(pil_image)
        else:
            # return white image if no image found
            patch = np.ones((512, 512, 3), dtype=np.uint8) * 255
        cursor.close()

    return patch

def plot_wsi_with_quad_tree(wsi: np.ndarray, resolution_level: int, plot_path: str, dpi: int=300):
    """ Plot the whole slide image (WSI) with a quad tree overlay.

    Args:
        wsi (np.ndarray): The WSI as a NumPy array.
        resolution_level (int): Resolution level of the WSI.
        plot_path (str): Path to the directory to save the plot in.
        dpi (int): Dots per inch for the plot. (default: 300)
    """
    side_length = wsi.shape[0]

    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(wsi)

    # Define the big level 1 grid 
    level = 1
    fontsize = plt.rcParams['font.size']
    add_subgrid(ax, wsi, level, (0, 0, 0), fontsize+1)

    # define smaller level 2 grids for top left and top right 
    level = 2
    add_subgrid(ax, wsi, level, (1, 0, 0), fontsize+1)
    # add_subgrid(ax, patch, level, (1, 0, 1))

    # define smaller level 3 grids for top left and left center 
    level = 3
    add_subgrid(ax, wsi, level, (2, 0, 0), fontsize-1.5)
    add_subgrid(ax, wsi, level, (2, 1, 1), fontsize-1.5)

    # add (0,0,0) label in center of the plot
    ax.text(side_length // 2, side_length // 2, f'({0},{0},{0})', color='black', ha='center', va='center', fontsize=fontsize+2)

    # Adjust plot limits and remove ticks
    ax.set_xlim([0, side_length])
    ax.set_ylim([side_length, 0])
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    path_to_plot = os.path.join(plot_path, f"patchcraft_quad_tree_level_{resolution_level}_dpi_{dpi}.pdf")
    plt.savefig(path_to_plot, format="pdf", bbox_inches="tight", dpi=dpi)
    #plt.show()


def add_subgrid(ax: plt.Axes, patch: np.ndarray, level: int, grid_loc: tuple, fontsize: int):
    """ Add a subgrid to the plot.

    Args:
        ax (plt.Axes): The plot axes.
        patch (np.ndarray): The patch as a NumPy array.
        level (int): Resolution level of the subgrid.
        grid_loc (tuple): Location of the subgrid in the format (level, x, y).
        fontsize (int): Font size for the block labels. (default: 16)
    """
    side_len = patch.shape[0]
    lvl_diff = level - grid_loc[0]
    multiplier = 2**lvl_diff
    coords = [(multiplier * grid_loc[1] + i, multiplier * grid_loc[2] + j) for j in range(multiplier) for i in range(multiplier)]

    step_size = side_len // 2**level
    x_min = min([c[0] for c in coords]) * step_size
    x_max = (max([c[0] for c in coords]) + 1) * step_size
    y_min = min([c[1] for c in coords]) * step_size
    y_max = (max([c[1] for c in coords]) + 1) * step_size

    # Add horizontal and vertical lines
    for x in range(min([c[0] for c in coords]), max([c[0] for c in coords])):
        ax.axvline(x=(x+1) * step_size, 
                   ymin=1 - y_min/side_len, 
                   ymax=1 - y_max/side_len, 
                   color='red', 
                   linestyle='--', 
                   linewidth=0.6)
    for y in range(min([c[1] for c in coords]), max([c[1] for c in coords])):
        ax.axhline(y=(y+1) * step_size, 
                   xmin=x_min/side_len, 
                   xmax=x_max/side_len, 
                   color='red', 
                   linestyle='--', 
                   linewidth=0.6)
        
    # Add block labels (level, x, y)
    for y in range(min([c[1] for c in coords]), max([c[1] for c in coords]) + 1):
        for x in range(min([c[0] for c in coords]), max([c[0] for c in coords]) + 1):
            ax.text( 
                x * step_size + step_size // 2, 
                y * step_size + step_size // 2,
                f'({level},{x},{y})', 
                color='black', 
                fontsize=fontsize, 
                ha='center', 
                va='center'
            )

def plot_tiles_with_patches(patch: np.ndarray, level: int, x_tile: int, y_tile: int, 
                            multiplier: int, plot_path: str):
    """ Plot a patch (region of the WSI) with grid lines and smaller patches.

    Args:
        patch (np.ndarray): The patch as a NumPy array.
        level (int): Resolution level of the patch.
        x_tile (int): x-coordinate of the patch.
        y_tile (int): y-coordinate of the patch.
        multiplier (int): Multiplier for the patch size.
        plot_path (str): Path to the directory to save the plot in.
    """
    # Get the side length of the patch and the step size for the grid
    side_len = patch.shape[0]
    step_size = side_len // multiplier

    # Plot setup
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(patch)

    # Add a big patch
    patch_size = 200 * multiplier
    patch_x = 210 * multiplier
    patch_y = 260 * multiplier
    patch = patches.Rectangle(
        (patch_x, patch_y), patch_size, patch_size, edgecolor="black", facecolor="green", alpha=0.5, label="Patch"
    )
    ax.add_patch(patch)

    # Add a small patch
    small_patch_size = 50 * multiplier
    small_patch_x = 390 * multiplier
    small_patch_y = 133 * multiplier
    fontsize_small_patch = plt.rcParams['font.size'] - 2
    small_patch = patches.Rectangle(
        (small_patch_x, small_patch_y), small_patch_size, small_patch_size, linewidth=1, edgecolor="black", facecolor="green", alpha=0.5, label="Patch"
    )
    ax.add_patch(small_patch)

    lw = 0.6
    for x in range(step_size, side_len, step_size):
        # Line outside the patch
        if x < patch_x or x > patch_x + patch_size:
            plt.axvline(x=x, color='black', linestyle='-', linewidth=lw)
        else:
            # Line segment above the patch
            plt.plot([x, x], [0, patch_y], color='black', linestyle='-', linewidth=lw)
            # Line segment inside the patch
            plt.plot([x, x], [patch_y, patch_y + patch_size], color='black', linestyle='--', linewidth=lw)
            # Line segment below the patch
            plt.plot([x, x], [patch_y + patch_size, side_len], color='black', linestyle='-', linewidth=lw)

    # Add horizontal grid lines
    for y in range(step_size, side_len, step_size):
        # Line outside the patch
        if y < patch_y or y > patch_y + patch_size:
            plt.axhline(y=y, color='black', linestyle='-', linewidth=lw)
        else:
            # Line segment to the left of the patch
            plt.plot([0, patch_x], [y, y], color='black', linestyle='-', linewidth=lw)
            # Line segment inside the patch
            plt.plot([patch_x, patch_x + patch_size], [y, y], color='black', linestyle='--', linewidth=lw)
            # Line segment to the right of the patch
            plt.plot([patch_x + patch_size, side_len], [y, y], color='black', linestyle='-', linewidth=lw)

    # Add labels to both patches and one for a tile
    ax.text(patch_x + patch_size / 2, patch_y + patch_size / 2, "Patch", color="white", ha="center", va="center")
    ax.text(small_patch_x + small_patch_size / 2, small_patch_y + small_patch_size / 2, "Patch", color="white", ha="center", va="center", fontsize=fontsize_small_patch)
    ax.text(
        step_size*0.5, step_size*(7 + 0.5), "Tile", color="black", ha="center", va="center", fontsize=fontsize_small_patch + 2
    )

    # Customize the legend with square handles
    square_patch = patches.Rectangle((0, 0), 1, 1, facecolor="green", alpha=0.5, edgecolor="none")
    tile_patch = patches.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", linewidth=lw)
    handles = [
        square_patch,
        tile_patch,
    ]
    labels = ["Patches", "Tiles (grid)"]
    legend = ax.legend(handles, labels, loc="upper left", handleheight=1, handlelength=1, borderpad=0.7)

    # Adjust plot limits, remove ticks, and set aspect ratio
    ax.set_xlim([0, side_len])
    ax.set_ylim([side_len, 0])  # Flip y-axis for proper orientation
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    # Add axis spines for plot boundary
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    plt.tight_layout()
    path_to_plot = os.path.join(plot_path, f"patchcraft_patches_and_tiles_({level},{x_tile},{y_tile}).pdf")
    plt.savefig(path_to_plot, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()

def augment_patch_and_plot(patch: torch.Tensor, plot_path: str, level: int, x_tile: int, y_tile: int):
    """ Augment a patch and plot the original and augmented patches.

    Args:
        patch (torch.Tensor): The patch as a PyTorch tensor.
        plot_path (str): Path to the directory to save the plot in.
        level (int): Resolution level of the patch.
        x_tile (int): x-coordinate of the patch.
        y_tile (int): y-coordinate of the patch.
    """
    # transform to torch tensor
    patch = torch.from_numpy(np.transpose(patch, (2,0,1))) # pytorch wants (batch, color, height, width)

    # Define augmentations
    vflip = T.RandomVerticalFlip(p=0.5)
    hflip = T.RandomHorizontalFlip(p=0.5)
    brightness = 0.2
    contrast = 0.2
    saturation = 0.2
    hue = 0.05
    color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    # Define augmentations and apply them to the patch
    augmentations = [vflip, hflip, color_jitter]
    augmented_patches = apply_augmentations(patch, augmentations)

    # Prepare the plot
    n_patches = len(augmented_patches) + 1
    figsize = plt.rcParams["figure.figsize"]
    fig, axs = plt.subplots(1, n_patches, figsize=figsize)

    # Plot original patch
    plot_patch_and_configure_axes(patch, axs[0])   
    axs[0].set_xlabel("Original Patch")

    # Plot augmented patches
    for i, p in enumerate(augmented_patches, start=1):
        plot_patch_and_configure_axes(p, axs[i])
    
    # add label "Augmentations" under the third patch
    axs[2].set_xlabel("Examples for Augmentated Patches")

    # Finalize the plot and save it
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Ensure labels are fully visible
    path_to_plot = os.path.join(plot_path, f"augmented_patches_({level},{x_tile},{y_tile})_({brightness},{contrast},{saturation},{hue}).pdf")
    plt.savefig(path_to_plot, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()

def apply_augmentations(patch: torch.Tensor, augmentations: list) -> list:
    """ Apply a list of augmentations to a patch.

    Args:
        patch (torch.Tensor): The patch as a PyTorch tensor.
        augmentations (list): List of augmentations to apply.

    Returns:
        list: List of augmented patches as PyTorch tensors.
    """
    augmented_patches = []
    for _ in range(len(augmentations)):
        augmented_patch = patch
        for augmentation in augmentations:
            augmented_patch = augmentation(augmented_patch)
        augmented_patches.append(augmented_patch)
    return augmented_patches

def plot_patch_and_configure_axes(patch: torch.Tensor, ax: plt.Axes):
    """ Plot a patch and configure the axes.

    Args:
        patch (torch.Tensor): The patch as a PyTorch tensor.
        ax (plt.Axes): The plot axes.
    """
    ax.imshow(np.transpose(patch.numpy(), (1, 2, 0)))
    ax.set_xticks([])
    ax.set_yticks([])
    # change color of axis to white
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')

def main(tex_plot: bool):
    """ Main function for visualizing preprocessing with Pamly and Patchcraft.

    Args:
        tex_plot (bool): Whether to use LaTeX for text rendering.
    """
    # set seed for reproducibility
    torch.manual_seed(42)

    # setup different paths and plot directory
    path_to_data = "/Users/ferdinandtolkes/data"
    slide_name = "87351-2018-0-HE-DLBCL"
    path_to_slide = os.path.join(path_to_data, slide_name + ".sqlite")
    plot_path = f"/Users/ferdinandtolkes/code/lymphoma/figures/plots_{slide_name}"
    os.makedirs(plot_path, exist_ok=True)

    # sample not from lowest level such that resolution is not too terrible
    resolution_level = 6
    wsi = get_wsi(path_to_slide, resolution_level) 
    # Plot the WSI with quad tree overlay
    set_plotting_config(fontsize=8, aspect_ratio=1/1, width_fraction=0.5, text_usetex=tex_plot)
    plot_wsi_with_quad_tree(wsi, resolution_level, plot_path)

    # Plot region (l=5, x=6, y=2) of WSI and visualize notion of patches
    level = 5
    x_tile = 6
    y_tile = 2
    highest_level = 8
    
    patch = get_patch(highest_level, level, x_tile, y_tile, path_to_slide)
    multiplier = 2**(highest_level - level)
    set_plotting_config(fontsize=8, aspect_ratio=1/1, width_fraction=0.5, text_usetex=tex_plot)
    plot_tiles_with_patches(patch, level, x_tile, y_tile, multiplier, plot_path)
    
    # augment patch and plot
    set_plotting_config(fontsize=10, aspect_ratio=8/4, width_fraction=1.0, text_usetex=tex_plot)
    augment_patch_and_plot(patch, plot_path, level, x_tile, y_tile)
    


    

if __name__ == '__main__':
    args = parse_arguments()
    main(**args)

