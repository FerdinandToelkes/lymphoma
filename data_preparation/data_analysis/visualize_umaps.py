import matplotlib.pyplot as plt
import argparse
import seaborn as sns 
import pickle
import numpy as np

from matplotlib.lines import Line2D

from lymphoma.resnet_ddp.eval_model.plot_saved_results import set_plotting_config
from lymphoma.diagnosis_maps import LABELS_MAP_STRING_TO_INT
from lymphoma.data_preparation.data_analysis.generate_umaps import parse_datasets

# This script has to be run locally because the server does not have tex installed
"""
To be executed in the directory code:

For comparing two datasets:
python3 -m lymphoma.data_preparation.data_analysis.visualize_umaps --datasets=kiel,munich --unique_slides --classes=CLL,DLBCL,FL,LTDS,MCL --n_neighbors=10

For viewing one dataset:
python3 -m lymphoma.data_preparation.data_analysis.visualize_umaps --datasets=kiel --unique_slides --classes=CLL,DLBCL,FL,LTDS,MCL --n_neighbors=10 
"""




def parse_arguments() -> dict:
    """ Parse command line arguments. 
    
    Returns:
        dict: Dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('-ds', '--datasets', default='kiel', type=str, help='Name of the datasets seperated by commas, can be a combination of "kiel", "swiss_1", "swiss_2", "multiply", "munich" (default: kiel)')
    parser.add_argument('--classes', default='CLL,DLBCL,FL,HL,LTDS,MCL', type=str, help='Classes to be used for the analysis seperated by commas. (default: "CLL,DLBCL,FL,HL,LTDS,MCL")')
    parser.add_argument('--unique_slides', default=True, action='store_true', help='Whether to only take the zeroth slide of a patient, i.e. 80212-2018-0-HE-FL but not 80212-2018-1-HE-FL (default: False)')
    parser.add_argument('--legend_placement', default='automatic', type=str, help='Placement of the legend. (default: automatic)')
    parser.add_argument('--n_neighbors', default=15, type=int, help='Number of neighbors to use for UMAP. (default: 15)')
    parser.add_argument('--min_dist', default=0.1, type=float, help='Minimum distance to use for UMAP. (default: 0.1)')
    args = parser.parse_args()
    return dict(vars(args))

def get_and_append_marker(datasets: list, c: str, markers: list, marker_labels: list, idx: int) -> tuple:
    """ Get the marker for the class and append it to the marker_labels list with its corresponding label. 
    
    Args:
        datasets (list): List of datasets.
        c (str): Current class (can also contain dataset information).
        markers (list): List of markers.
        marker_labels (list): List of tuples containing the marker and the corresponding label.
        idx (int): Index specifying at which class we are.

    Returns:
        Tuple containing the updated marker_labels list and the specification for the marker.
    """
    if len(datasets) == 1:
        diagnose_int = LABELS_MAP_STRING_TO_INT[c] - 1 # because Unknown is not in the map
        marker = markers[diagnose_int]
        marker_labels.append((marker, c))
    else:
        marker = markers[idx]
        label = f"{c.split('_')[0].capitalize()} {c.split('_')[1]}"
        marker_labels.append((marker, label))
    return marker_labels, marker



def load_one_umap_data(patch_size: str, dataset: str, classes: str, unique_slides: bool, n_neighbors: int, min_dist: float) -> tuple:
    """Load the UMAP data for a single dataset.
    
    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset.
        classes (str): Classes to be used for the analysis separated by commas.
        unique_slides (bool): Whether to only take the zeroth slide of a patient.
        n_neighbors (int): Number of neighbors to use for UMAP.
        min_dist (float): Minimum distance to use for UMAP.

    Returns:
        Tuple containing the data, the path to the data and the suffix of the file
    """
    path = f"/Users/ferdinandtolkes/code/lymphoma/data_preparation/data_analysis/results/{patch_size}/{dataset}/umap_plots"
    # data will have format: {class: {slide1: embeddings1, slide2: embeddings2, ...}}
    suffix = f"{classes}_us={unique_slides}_nn={n_neighbors}_md={min_dist}"
    name = f"reduced_embeddings_{suffix}.pkl"
    data = pickle.load(open(f"{path}/{name}", "rb"))
    return data, path, suffix


def plot_all_classes_one_dataset(data: dict, datasets: list, legend_placement: str, path: str, suffix: str) -> None:
    """Plot the UMAP embeddings for all classes in one dataset.

    Args:
        data (dict): Dictionary containing the embeddings for each class as a dictionary with the slide names
                     as keys and the embeddings as concatenated tensors as values.
        datasets (list): List of datasets.
        legend_placement (str): Placement of the legend.
        path (str): Path to save the plot to.
        suffix (str): Suffix of the file.
    """
    # setup the plot
    set_plotting_config(fontsize=8, aspect_ratio=1.0, width_fraction=.5, text_usetex=True)
    fig = plt.figure()
    ax = fig.gca()
    # get the colors and markers for the plot
    colors, markers = get_colors_and_markers(data)
    
    offset = 0
    marker_labels = []
    # iterate over all slides and plot all patch embeddings
    for idx, (c, slides_dir) in enumerate(data.items()):
        # use different markers for each class
        marker_labels, marker = get_and_append_marker(datasets, c, markers, marker_labels, idx)
        
        for s_idx, (slide, embs) in enumerate(slides_dir.items()):
            plot_points_one_slide(ax, embs, colors, offset, s_idx, marker)
        offset += len(slides_dir)
        
    # finalize and save the plot
    finalize_single_plot(ax, marker_labels, legend_placement)
    save_and_show_plot(path, suffix)

def get_colors_and_markers(data: dict) -> tuple:
    """Get the colors and markers for the plot.

    Args:
        data (dict): Dictionary containing the embeddings for each class as a dictionary with the slide names 
                     as keys and the embeddings as concatenated tensors as values.

    Returns:
        Tuple containing the figure, axis, colors and markers
    """
    # Define a sufficiently large color palette
    num_colors = sum(len(slides_dir) for slides_dir in data.values())  # Total slides
    colors = sns.color_palette("husl", num_colors) 
    # randomly permute colors to avoid similar colors for similar classes
    np.random.shuffle(colors)
    markers = ['x', '+', '1', '2', (8,2,0), '3'] # (8,2,0) is a star

    return colors, markers

def plot_points_one_slide(ax: plt.Axes, embs: np.ndarray, colors: list, offset: int, s_idx: int, marker: str) -> None:
    """Plot the points for one slide.

    Args:
        ax (plt.Axes): Axis of the plot.
        embs (np.ndarray): Embeddings for the slide.
        colors (list): List of colors.
        offset (int): Offset for the colors.
        s_idx (int): Index of the slide.
        marker (str): Marker for the slide.
    """
    ax.scatter(embs[:, 0], embs[:, 1], s=0.1, color=colors[offset + s_idx], zorder=1)
    # mark median x and y position with some marker depending on idx
    median_x = np.median(embs[:, 0])
    median_y = np.median(embs[:, 1])
    ax.scatter(median_x, median_y, s=4, color="black", marker=marker, zorder=2)

def finalize_single_plot(ax: plt.Axes, marker_labels: list, legend_placement: str) -> None:
    """Finalize the plot by adding the legend and saving it to the path.

    Args:
        ax (plt.Axes): Axis of the plot.
        marker_labels (list): List of tuples containing the marker and the corresponding label.
        path (str): Path to save the plot to.
        suffix (str): Suffix of the file.
        legend_placement (str): Placement of the legend.
    """
    # add legend for associating markers with classes
    legend_labels = [l for _, l in marker_labels]

    if legend_placement == "automatic":
        ax.legend([plt.scatter([0], [0], s=4, color='black', marker=m) for m, _ in marker_labels], legend_labels, markerscale=2)
    else:
        ax.legend([plt.scatter([0], [0], s=4, color='black', marker=m) for m, _ in marker_labels], legend_labels, markerscale=2, loc=legend_placement)
    # disable the ticks
    ax.set_xticks([])
    ax.set_yticks([])

def finalize_single_plot_text_legend(ax: plt.Axes, c: str) -> None:
    """Finalize the plot by adding the legend and saving it to the path.

    Args:
        ax (plt.Axes): Axis of the plot.
        c (str): Current class.
    """
    # add legend that shows the class
    ax.text(0.05, 0.95, c, transform=ax.transAxes, verticalalignment='top')
    
    # disable the ticks
    ax.set_xticks([])
    ax.set_yticks([])


def save_and_show_plot(path: str, suffix: str) -> None:
    """Save the plot to the path.

    Args:
        path (str): Path to save the plot to.
        suffix (str): Suffix of the file.
    """
    plt.savefig(f"{path}/umap_{suffix}.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved plot to {path}/umap_{suffix}.pdf")

def finalize_grid_plot(axes: list, marker_labels: list) -> None:
    """Finalize the grid plot by adding the legend in the empty plot.

    Args:   
        axes (list): List of axes of the plots.
        marker_labels (list): List of tuples containing the marker and the corresponding
                                label for the legend.
    """
    # Create legend markers without plotting
    legend_handles = [Line2D([0], [0], marker=m, color='black', markersize=2, linestyle='None') for m, _ in marker_labels]
    legend_labels = [f"{l.split(' ')[0]} slides" for _, l in marker_labels]
    # Position the legend inside the last empty plot
    axes[5].legend(legend_handles, legend_labels, loc='upper left', markerscale=2)
    # Remove the axis of the last plot to keep it clean
    axes[5].axis('off')
    plt.tight_layout()

def main(patch_size: str, datasets: str, classes: str, unique_slides: bool, legend_placement: str, n_neighbors: int, min_dist: float):
    """Main function to visualize the UMAP embeddings.
    
    Args:
        patch_size (str): Size of the patches.
        datasets (str): Name of the datasets seperated by commas.
        classes (str): Classes to be used for the analysis seperated by commas.
        unique_slides (bool): Whether to only take the zeroth slide of a patient.
        legend_placement (str): Placement of the legend.
        n_neighbors (int): Number of neighbors to use for UMAP.
        min_dist (float): Minimum distance to use for UMAP.
    """
    # load the data from results/umap_plots/
    datasets = parse_datasets(datasets)
    # plot either all classes in one plot 
    if len(datasets) == 1:
        dataset = datasets[0]
        data, path, suffix = load_one_umap_data(patch_size, dataset, classes, unique_slides, n_neighbors, min_dist)
        plot_all_classes_one_dataset(data, datasets, legend_placement, path, suffix)
    # or six classes in a 2x3 grid 
    else:
        # prepare the plot
        set_plotting_config(fontsize=8, aspect_ratio=3/2, width_fraction=1, text_usetex=True)
        # Create a 2x3 grid of subplots
        fig = plt.figure()
        axes = [fig.add_subplot(2, 3, i + 1) for i in range(6)] 
        
        dataset = "_".join(datasets)
        classes_list = classes.split(',')
        for i, c in enumerate(classes_list):
            data, path, suffix = load_one_umap_data(patch_size, dataset, c, unique_slides, n_neighbors, min_dist) 
            colors, markers = get_colors_and_markers(data)

            offset = 0
            marker_labels = []
            for idx, (ds_and_cls, slides_dir) in enumerate(data.items()):
                # use different markers for each class
                marker_labels, marker = get_and_append_marker(datasets, ds_and_cls, markers, marker_labels, idx)
                
                for s_idx, (slide, embs) in enumerate(slides_dir.items()):
                    plot_points_one_slide(axes[i], embs, colors, offset, s_idx, marker)
                offset += len(slides_dir)
                
            # finalize the plot
            finalize_single_plot_text_legend(axes[i], c)
            
        finalize_grid_plot(axes, marker_labels)
        
        # save the plot
        updated_suffix = f"{classes}_us={unique_slides}_nn={n_neighbors}_md={min_dist}"
        save_and_show_plot(path, updated_suffix)
            
       

if __name__ == "__main__":
    args = parse_arguments()
    main(**args)
