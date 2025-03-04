import matplotlib.pyplot as plt
import argparse

import torch
import numpy as np
import os

from lymphoma.resnet_ddp.eval_model.plot_saved_results import set_plotting_config


# Example usage:
"""
docker run --shm-size=400gb --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code:/mnt -v /home/ftoelkes/preprocessed_data/train_data:/data ftoelkes_lymphoma python3 -m lymphoma.data_preparation.data_analysis.visualize
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





def main(patch_size: str, dataset: str):
    """ Visualize some exemplary patches. 
    
    Args:
        patch_size (str): Size of the patches. 
        dataset (str): Name of the dataset, can be "kiel" for example.
    """
    data_path = f"/data/{patch_size}/{dataset}/patches/data_dir"
    patches = []
    np.random.seed(42)
    for c in os.listdir(data_path):
        class_dir = os.path.join(data_path, c)
        slides_in_class = os.listdir(class_dir)
        np.random.shuffle(slides_in_class)

        for slide in slides_in_class[:6]:
            slide_path = os.path.join(class_dir, slide)
            patch_paths = os.listdir(slide_path)
            np.random.shuffle(patch_paths)
            # load some patches 
            patch = torch.load(os.path.join(slide_path, patch_paths[0]))
            patches.append(patch)
    print(f"len(patches): {len(patches)}")

    # plot patches on a 6x6 grid
    set_plotting_config(fontsize=10, aspect_ratio=1.0, width_fraction=1.0, text_usetex=True)
    fig, axes = plt.subplots(6, 6)
    for i, patch in enumerate(patches):
        ax = axes[i % 6, i // 6]
        ax.imshow(patch[0].numpy().transpose(1, 2, 0))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("/mnt/patches.pdf", format='pdf', dpi=300)

    
       

if __name__ == "__main__":
    args = parse_arguments()
    main(**args)
    