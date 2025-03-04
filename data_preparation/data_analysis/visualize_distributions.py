import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns


from lymphoma.resnet_ddp.eval_model.plot_saved_results import set_plotting_config
# this script has similar purpose as analyze_data_distributions.py, but originated after switching to using multiple data sets, i.e. its newer

# This script has to be run locally because the server does not have tex installed
"""
To be executed in the directory code:

python3 -m lymphoma.data_preparation.data_analysis.visualize_distributions --dataset kiel
"""





def parse_arguments() -> dict:
    """ Parse command line arguments. 
    
    Returns:
        dict: Dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply", "munich" or "all_data". (default: kiel)')
    args = parser.parse_args()
    return dict(vars(args))

def load_data_and_sum_over_data_dirs(path: str, name: str) -> dict:
    """ Load the data from the csv file in the given path and sum over the columns to 
    get the total number of data points per class.

    Args:
        path (str): path to the csv file
        name (str): name of the csv file

    Returns:
        dict: dictionary containing the total number of data points per class
    """
    df = pd.read_csv(f"{path}/{name}.csv", index_col=0)
    return df.sum(axis=1).to_dict()

def plot_total_number_of_data_points(data: dict, path: str, name: str, dataset: str, extra_y_padding: int = 0):
    """ Plot the total number of data points per class without the 'total' key.

    Args:
        data (dict): dictionary containing the number of data points per class
        path (str): path to save the plot to
        name (str): name of the plot
        dataset (str): name of the dataset for the title
        extra_y_padding (int, optional): extra padding for the y-axis
    """
    # Remove the 'total' key and all keys with value 0
    data.pop("total")
    data = {k: v for k, v in data.items() if v != 0}

    # Set plotting configuration and professional color palette
    set_plotting_config(fontsize=8, aspect_ratio=1.0, width_fraction=.5, text_usetex=True)
    color = sns.color_palette("muted")[0]

    # Create bar plot
    fig = plt.figure()
    ax = fig.gca()
    bars = ax.bar(
        data.keys(),
        data.values(),
        color=color,
        edgecolor='black'
    )

    # Add total value labels
    padding_factor = 0.01
    for x, total in zip(data.keys(), data.values()):
        ax.text(
            x, total + (padding_factor * max(data.values())),  # Slightly above the bar
            f"{int(total)}",
            ha='center',
            va='bottom',
        )
    # add extra padding for the y-axis
    y_padding = max(data.values()) * padding_factor 
    ax.set_ylim(0, max(data.values()) + (7+extra_y_padding)*y_padding)

    # Set titles and labels
    ax.set_xlabel("Class")
    ylabel = "Number of Slides" if "slides" in name else "Number of Patches"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset.capitalize()} Dataset")

    # Optimize layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{path}/{name}.pdf", format='pdf')  # Save as high-res PDF
    plt.show()


def plot_unique_and_total_number_of_data_points(unique_data: dict, data: dict, path: str, name: str, dataset: str):
    """
    Plot the unique and total number of data points per class as stacked bars, with unique slides
    represented by a darker shaded region.

    Args:
        unique_data (dict): dictionary containing the number of unique data points per class
        data (dict): dictionary containing the number of data points per class
        path (str): path to save the plot to
        name (str): name of the plot
        dataset (str): name of the dataset for the title
    """
    # Remove the 'total' key and keys with value 0
    unique_data = {k: v for k, v in unique_data.items() if v != 0 and k != "total"}
    data = {k: v for k, v in data.items() if v != 0 and k != "total"}

    differences_data = {k: data[k] - unique_data[k] for k in data.keys()}

   
    # Colors for unique and total slides
    colors = sns.color_palette("muted", n_colors=2)

    # Prepare the plot
    set_plotting_config(fontsize=8, aspect_ratio=1.0, width_fraction=.5, text_usetex=True)
    fig = plt.figure()
    ax = fig.gca()

    # Create stacked bar plot
    bottom = list(unique_data.values())
    total_bars = ax.bar(
        data.keys(),
        differences_data.values(),
        bottom=bottom,
        color=colors[1],
        label='All Slides',
        edgecolor='none',
    )

    unique_bars = ax.bar(
        data.keys(),
        unique_data.values(),
        color=colors[0],
        label='Unique Slides',
        edgecolor='none',  # Black edges for the sides and top
    )

    # Add the top and side edges for total bars manually
    for total_bar, unique_bar in zip(total_bars, unique_bars):
        # Get the coordinates of the bar
        x = total_bar.get_x()
        y = unique_bar.get_y() 
        width = total_bar.get_width()
        height = total_bar.get_height() + unique_bar.get_height()
        
        # Draw the top and side edges using `Rectangle`
        rect = plt.Rectangle(
            (x, y),  # Start from the bottom of the bar
            width,
            height,
            facecolor='none',  # No fill
            edgecolor='black',  # Black edge
        )
        ax.add_patch(rect)

    # Add total value labels
    padding_factor = 0.01
    for x, total in zip(data.keys(), data.values()):
        ax.text(
            x, total + (padding_factor * max(data.values())),  # Slightly above the bar
            f"{int(total)}",
            ha='center',
            va='bottom',
        )

    # Add unique value labels
    for x, unique in zip(unique_data.keys(), unique_data.values()):
        # Add the unique value below the bar in white text
        ax.text(
            x, unique - 2*(padding_factor * max(data.values())),  # Slightly below the bar
            f"{int(unique)}",
            ha='center',
            va='top',
            color='white',
        )
        
    
    # add extra padding for the y-axis
    extra_y_padding = max(data.values()) * padding_factor
    ax.set_ylim(0, max(data.values()) + 7*extra_y_padding)

    # Set axis labels and title
    ax.set_xlabel("Class")
    ylabel = "Number of Slides" if "slides" in name else "Number of Patches"
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(data)))
    ax.set_title(f"{dataset.capitalize()} Dataset")

    # Add legend
    ax.legend()

    # Optimize layout and save the plot
    plt.tight_layout()
    plt.savefig(f"{path}/{name}.pdf", format='pdf')  # Save as high-res PDF
    plt.show()
    print(f"Saved plot to {path}/{name}.pdf")



def main(patch_size: str, dataset: str):
    """ Visualize number of unique and total slides and patches per class.
    
    Args:
        patch_size (str): Size of the patches. 
        dataset (str): Name of the dataset, e.g. "kiel"
    """
    # save results to file and print them
    path = f"lymphoma/data_preparation/data_analysis/results/{patch_size}/{dataset}/distributions"
    total_unique_slides_per_class = load_data_and_sum_over_data_dirs(path, "unique_slides")
    total_unique_patches_per_class = load_data_and_sum_over_data_dirs(path, "unique_patches")
    total_slides_per_class = load_data_and_sum_over_data_dirs(path, "slides")
    total_patches_per_class = load_data_and_sum_over_data_dirs(path, "patches")
    

    # plot total number of slides and patches per class
    # plot_total_number_of_data_points(total_unique_slides_per_class, path, "total_number_of_unique_slides_per_class")
    extra_y_padding = 8 if dataset == "munich" else 0
    plot_total_number_of_data_points(total_patches_per_class, path, "total_number_of_patches_per_class", dataset, extra_y_padding)
    plot_unique_and_total_number_of_data_points(total_unique_slides_per_class, total_slides_per_class,
                                                path, "unique_and_total_number_of_slides_per_class", dataset)
    
    
    
       

if __name__ == "__main__":
    args = parse_arguments()
    main(**args)
    