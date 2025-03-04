import os
import torch
import time
import argparse
import numpy as np
import torchvision.transforms.v2 as T 

from tqdm import tqdm

from data_preparation.embedding_generation.generate_uni_embeddings import load_uni_model, prepare_attention_maps
from data_preparation.embedding_generation.custom_vision_transformer import load_wrapper_model 


"""
screen -dmS generate_toy_data sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code/lymphoma:/mnt -v /home/ftoelkes/preprocessed_data/test_data:/data -v /home/ftoelkes/code/lymphoma/vit_large_patch16_256.dinov2.uni_mass100k:/model ftoelkes_lymphoma python3 -m uni_analysis.generate_toy_data -ps "1024um" -ds "kiel" -ts 512 ; exec bash'
"""


def parse_arguments() -> dict:
    """ Parse the arguments for the embedding generation

    Returns:
        dict: Dictionary with the arguments
    """
    parser = argparse.ArgumentParser(description='Generate embeddings and attention maps from exemplary patches.')
    # general arguments
    parser.add_argument('--target_dir', default="test_embeddings_dir", type=str, help='Directory to save the embeddings and attentions to. (default: test_embeddings_dir)')
    parser.add_argument('-ps', '--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply" or "munich". (default: kiel)')
    parser.add_argument('--nr_of_patches', default=15, type=int, help='Number of patches to consider per slide. (default: 15)')
    # arguments for the embedding and attention generation
    parser.add_argument('-ts', '--target_size', default=512, type=int, help='Target size to which the patches are resized to. (default: 512)')
    parser.add_argument('-cs', '--crop_size', default=3072, type=int, help='Size of the randomly cropped image. (default: 3072)')
    parser.add_argument('-sa', '--save_attentions', default=False, action='store_true', help='Whether to save the attention maps. (default: False)')
    parser.add_argument('-hf', '--enable_hflip', default=False, action='store_true', help='Whether to enable horizontal flip. (default: False)')
    parser.add_argument('-vf', '--enable_vflip', default=False, action='store_true', help='Whether to enable vertical flip. (default: False)')
    parser.add_argument('-ec', '--enable_crop', default=False, action='store_true', help='Whether to enable cropping. (default: False)')
    parser.add_argument('-cj', '--color_jitter', default=False, action='store_true', help='Whether to enable color jitter. (default: False)')

    args = parser.parse_args()
    return dict(vars(args))

####################################################################################################################################
################################################## Embedding generation functions ##################################################
####################################################################################################################################

def get_slide_paths(path_to_class_dirs: str) -> list:
    """ Get the paths to the slides in the class directories. 
    
    Args:
        path_to_class_dirs (str): Path to the class directories

    Returns:    
        list: List with the paths to the slides
    """
    path_to_slides = []
    # get 1 slide paths per class
    for c in os.listdir(path_to_class_dirs):
        class_dir = os.path.join(path_to_class_dirs, c)
        slides_in_class = os.listdir(class_dir)
        slides_in_class.sort()
        path_to_slides.append(os.path.join(class_dir, slides_in_class[0]))
        # path_to_slides.extend([os.path.join(class_dir, slide) for slide in slides_in_class]) 
    print(f"Took {len(path_to_slides)} slides from {path_to_class_dirs}")
    path_to_slides.sort()
    return path_to_slides

def setup_for_embedding_generation(target_dir: str, patch_size: str, dataset: str, 
                                   target_size: int, save_attentions: bool) -> tuple:
    """ Setup the paths to the slides, the embedding directory, the model and the device. 
    
    Args:
        target_dir (str): Target directory for saving the embeddings
        patch_size (str): Size of the patches
        dataset (str): Name of the dataset
        target_size (int): Target size for resizing input images
        save_attentions (bool): Whether to save the attention maps

    Returns:
        tuple: Tuple with the paths to the slides, the path to the embedding directory, the model and the device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_dir = "/model"
    base_path = os.path.join("/data", patch_size, dataset, "patches")
    if not save_attentions:
        model = load_uni_model(device, local_dir=local_dir)
        path_to_embedding_dir = os.path.join(base_path, target_dir, f"{target_size}")
        print("Model that does not save attention maps loaded")
    else:
        model = load_wrapper_model(device, local_dir)
        path_to_embedding_dir = os.path.join(base_path, target_dir, f"{target_size}_attentions")
        print("Model that saves attention maps loaded")
    
    path_to_class_dirs = os.path.join(base_path, "data_dir")
    path_to_slides = get_slide_paths(path_to_class_dirs)
    return path_to_slides, path_to_embedding_dir, model, device

def generate_embeddings(path_to_slides: list, path_to_embedding_dir: str, model: torch.nn.Module, 
                        device: torch.device, target_size: int, nr_of_patches: int, 
                        save_attentions: bool, enable_hflip: bool, enable_vflip: bool, enable_crop: bool, 
                        crop_size: int, color_jitter: bool) -> str:
    """ Generate embeddings for the patches of the slides in path_to_slides and save them to path_to_embedding_dir. 
    
    Args:
        path_to_slides (list): List with the paths to the slides
        path_to_embedding_dir (str): Root path to the directory where the embeddings shall be saved
        model (torch.nn.Module): Model for generating the embeddings
        device (torch.device): Device where the model is loaded
        target_size (int): Target size for resizing the input images
        nr_of_patches (int): Number of patches to consider per slide
        save_attentions (bool): Whether to save the attention maps
        enable_hflip (bool): Whether to enable horizontal flip
        enable_vflip (bool): Whether to enable vertical flip
        enable_crop (bool): Whether to enable cropping
        crop_size (int): Size of the randomly cropped image
        color_jitter (bool): Whether to enable color jitter

    Returns:
        str: Path to the directory where the embeddings are saved
    """
    # set numpy seed
    np.random.seed(42)
    path_to_embedding_dir = augmentation_setup(path_to_embedding_dir, enable_hflip, enable_vflip, 
                                               enable_crop, crop_size, color_jitter)

    for slide_path in tqdm(path_to_slides, total=len(path_to_slides)):
        print(f"Processing slide {slide_path}")
        mean, std = torch.load(os.path.join(slide_path, "mean_std.pt"))

        # we only need last two directories of the slide path
        # slide path is of the form /data/1024um/kiel/patches/data_dir/FL/80212-2018-0-HE-FL
        slide_path_end = os.path.join(*slide_path.split("/")[-2:]) # * unpacks the list
        if not os.path.exists(os.path.join(path_to_embedding_dir, slide_path_end)):
            print(f"Creating directory {path_to_embedding_dir}/{slide_path_end}")
            os.makedirs(os.path.join(path_to_embedding_dir, slide_path_end), exist_ok=False) # raise error if directory already exists
         
        transform = T.Compose([            
            T.Normalize(mean, std),
            T.Resize((target_size, target_size), antialias=True) 
        ])
        # get the patches, shuffle them and take the first nr_of_patches
        patches = os.listdir(slide_path)
        patches = [patch for patch in patches if patch.startswith("patch") and patch.endswith(".pt")]
        np.random.shuffle(patches)
        patches = patches[:nr_of_patches]

        for patch in tqdm(patches, total=len(patches), desc=f"Generating embeddings for slide at {slide_path}"):
            name = os.path.join(path_to_embedding_dir, slide_path_end, patch)
            attentions_name = name.split(".pt")[0] + "_attentions.pt"
            if os.path.exists(name) and os.path.exists(attentions_name):
                continue
            image, label = torch.load(os.path.join(slide_path, patch))
            image = augment_image(image, enable_hflip, enable_vflip, enable_crop, crop_size, color_jitter)
            image = transform(image.div(255)).unsqueeze(0).to(device) # since model expects batch dimension

            with torch.no_grad():
                embedding = model(image).squeeze().detach().cpu() # remove batch dimension again
            torch.save((embedding,label), name)
            
            if save_attentions:
                image_size = image.shape[-1]
                attentions = prepare_attention_maps(model, image_size)
                torch.save(attentions, attentions_name)
    return path_to_embedding_dir

def augmentation_setup(path_to_embedding_dir: str, enable_hflip: bool, enable_vflip: bool, 
                       enable_crop: bool, crop_size: int, color_jitter: bool) -> str:
    """ Setup the path to the embedding directory according to the augmentations. 
    
    Args:
        path_to_embedding_dir (str): Path to the embedding directory
        enable_hflip (bool): Whether to enable horizontal flip
        enable_vflip (bool): Whether to enable vertical flip
        enable_crop (bool): Whether to enable cropping
        crop_size (int): Size of the randomly cropped image
        color_jitter (bool): Whether to enable color jitter

    Returns:
        str: Path to the embedding directory
    """
    if enable_hflip:
        path_to_embedding_dir += "_hflip"
        print(f"Horizontal flip enabled. Saving embeddings to {path_to_embedding_dir}")
    if enable_vflip:
        path_to_embedding_dir += "_vflip"
        print(f"Vertical flip enabled. Saving embeddings to {path_to_embedding_dir}")
    if enable_crop:
        path_to_embedding_dir += f"_crop_{crop_size}"
        print(f"Crop enabled. Saving embeddings to {path_to_embedding_dir}")
    if color_jitter:
        path_to_embedding_dir += "_color_jitter"
        print(f"Color jitter enabled. Saving embeddings to {path_to_embedding_dir}")
    return path_to_embedding_dir

def augment_image(image: torch.Tensor, enable_hflip: bool, enable_vflip: bool, 
                  enable_crop: bool, crop_size: int, color_jitter: bool) -> torch.Tensor:
    """ Augment the image with horizontal flip, vertical flip, random crop and color jitter. 
    
    Args:
        image (torch.Tensor): Image to augment
        enable_hflip (bool): Whether to enable horizontal flip
        enable_vflip (bool): Whether to enable vertical flip
        enable_crop (bool): Whether to enable cropping
        crop_size (int): Size of the randomly cropped image
        color_jitter (bool): Whether to enable color jitter

    Returns:
        torch.Tensor: Augmented image
    """
    if enable_hflip:
        image = image.flip(1) # horizontal flip 
    if enable_vflip:
        image = image.flip(2) # vertical flip
    if enable_crop:
        image = T.RandomCrop((crop_size, crop_size))(image)
    if color_jitter:
        image = random_color_jitter(image)
    return image

def random_color_jitter(tensor: torch.Tensor) -> torch.Tensor:
    # Function taken from patchcraft/src/patchcraft/sample_tiles/augment.py and values taken form config.yaml"""
    """Randomly change the brightness, contrast, saturation and hue of the image. 
    
    Args:
        tensor (torch.Tensor): Image to augment

    Returns:
        torch.Tensor: Image with applied color jitter
    """
    jitter = T.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.025)
    return jitter(tensor)

def save_execution_time(path_to_embedding_dir: str, execution_time: float):
    """ Save the execution time to a txt file. 
    
    Args:
        path_to_embedding_dir (str): Path to the embedding directory
        execution_time (float): Execution time in seconds
    """
    # create txt file to save timings if it does not exist
    path_to_parent_dir = os.path.dirname(path_to_embedding_dir)
    path_to_timing_file = os.path.join(path_to_parent_dir, "timings.txt")
    if not os.path.exists(path_to_timing_file):
        with open(path_to_timing_file, "w") as f:
            f.write("Variant, Execution Time (seconds)\n")
    # also save the execution time to the directory where results of analysis will be saved
    if not os.path.exists(os.path.join("/mnt/uni_analysis/results", "timings.txt")):
        with open(os.path.join("/mnt/uni_analysis/results", "timings.txt"), "w") as f:
            f.write(f"{execution_time:.2f}")
    # append the execution time to the timings file
    with open(path_to_timing_file, "a") as f:
        f.write(f"{path_to_embedding_dir.split('/')[-1]}: {execution_time:.2f}\n")
    print(f"Execution time: {execution_time} seconds appended to {path_to_timing_file}")


####################################################################################################################


def main(target_dir: str, patch_size: str, dataset: str, nr_of_patches: int, 
         save_attentions: bool, target_size: int, enable_hflip: bool, enable_vflip: bool, 
         enable_crop: bool, crop_size: int, color_jitter: bool):
    """ Main function for the embedding and attention generation.

    Args:
        target_dir (str): Target directory for saving the embeddings
        patch_size (str): Size of the patches
        dataset (str): Name of the dataset
        nr_of_patches (int): Number of patches to consider per slide
        save_attentions (bool): Whether to save the attention maps
        target_size (int): Target size for resizing the input images
        enable_hflip (bool): Whether to enable horizontal flip
        enable_vflip (bool): Whether to enable vertical flip
        enable_crop (bool): Whether to enable cropping
        crop_size (int): Size of the randomly cropped image
        color_jitter (bool): Whether to enable color jitter
    """
    # setup everything for the embedding generation
    print("Generating embeddings")
    path_to_slides, path_to_embedding_dir, model, device = setup_for_embedding_generation(target_dir, patch_size, dataset, target_size, save_attentions)

    if os.path.exists(path_to_embedding_dir):
        raise ValueError(f"Directory {path_to_embedding_dir} already exists. Will not overwrite it.")

    start = time.time()
    path_to_embedding_dir = generate_embeddings(path_to_slides, path_to_embedding_dir, model, device, target_size, 
                            nr_of_patches, save_attentions, enable_hflip, enable_vflip, enable_crop, crop_size, color_jitter)
    execution_time = time.time() - start
    save_execution_time(path_to_embedding_dir, execution_time)
    

if __name__ == '__main__':
    args = parse_arguments()
    main(**args)   
    

	
