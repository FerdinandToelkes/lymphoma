import os
import torch
import argparse
import time
import numpy as np
import torchvision.transforms.v2 as T 

from torchvision import transforms
from tqdm import tqdm

# own modules
from .generate_uni_embeddings import get_slide_paths, make_class_directories, load_uni_model

# run this script using the helper bash script run_gen_and_concat_top_embs.sh

def parse_arguments() -> dict:
    """ Parse command line arguments. 
    
    Returns:
        dict: Dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Prepare normalization of patches.')
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply" or "munich". (default: kiel)')
    parser.add_argument('--starting_slide', default=0, type=int, help='Index of the first slide to start with. (default: 0)')
    parser.add_argument('--nr_of_slides', default=1, type=int, help='Number of slides to process. (default: 1)')
    parser.add_argument('--nr_of_attended_regions', default=1, type=int, help='Number of maximal attention values to compare between each map. (default: 1)')
    parser.add_argument('-ao', '--allow_overlap', default=False, action='store_true', help='Allow overlapping regions in the top n attended regions. (default: False, if flag is set: True)')
    args = parser.parse_args()
    return vars(args)

def general_setup(patch_size: str, dataset: str, nr_of_attended_regions: int, allow_overlap: bool) -> tuple:
    """ Setup the model, device, path to class directories and path to embedding and attention directory. 
    
    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset.
        nr_of_attended_regions (int): Number of maximal attention values to compare between each map.
        allow_overlap (bool): Allow overlapping regions in the top n attended regions.

    Returns:
        tuple: Tuple containing the model, device, path to class directories
               and path to embedding and attention directory.
    """
    torch.hub.set_dir('tmp/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    model = load_uni_model(device, local_dir="/model")
    print("model loaded")
   
    path_to_class_dirs = os.path.join("/data", patch_size, dataset, "patches", "data_dir")
    path_to_embedding_dir = create_base_and_class_directories(patch_size, dataset, nr_of_attended_regions, allow_overlap)
    path_to_attention_dir = os.path.join("/data", patch_size, dataset, "patches", "attentions_dir")

    return model, device, path_to_class_dirs, path_to_embedding_dir, path_to_attention_dir

def create_base_and_class_directories(patch_size: str, dataset: str, nr_of_attended_regions: int, 
                                      allow_overlap: bool) -> str:
    """ Create the base directory for the concatenated embeddings of the top n attended regions and 
    the class directories in which the directories with the corresponding slide name will be created in.

    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset.
        nr_of_attended_regions (int): Number of maximal attention values to compare between each map.
        allow_overlap (bool): Allow overlapping regions in the top n attended regions.

    Returns:
        str: Path to the directory containing the concatenated embeddings of the top n attended regions.     
    """
    # setup path to directory with different classes containing the slide directories with the patches
    base_path = os.path.join("/data", patch_size, dataset, f"top_{nr_of_attended_regions}")
    if allow_overlap:
        target_dir = base_path + "_overlap_patches"
    else:        
        target_dir = base_path + "_patches"
    path_to_embedding_dir = os.path.join(target_dir, "embeddings_dir")

    if not os.path.exists(path_to_embedding_dir):
        print(f"Creating directory {path_to_embedding_dir} and class directories")
        os.makedirs(path_to_embedding_dir, exist_ok=True) # create directory if it does not exist
        make_class_directories(path_to_embedding_dir)

    return path_to_embedding_dir


def prepare_for_patch_iteration(slide_path: str, path_to_embedding_dir: str) -> tuple:
    """ Prepare for iteration over the patches in the slide directory 
        and create the target directories for the concatenated embeddings. 
    
    Args:
        slide_path (str): Path to the slide directory.
        path_to_embedding_dir (str): Path to the directory containing the concatenated embeddings of the top n attended regions.

    Returns:
        tuple: Tuple containing the transform, the end of the slide path and the patches in the slide directory.
    """  
    # load mean and std for normalization
    mean, std = torch.load(os.path.join(slide_path, "mean_std.pt"))
    # we only need last two directories of the slide path
    # slide path is of the form /data/400um_patches/data_dir/FL/80212-2018-0-HE-FL
    slide_path_end = os.path.join(*slide_path.split("/")[-2:]) # * unpacks the list
    create_slide_directories(path_to_embedding_dir, slide_path_end)

    transform = transforms.Compose([            
        transforms.Normalize(mean, std)
    ])
    # get all patches in the slide directory
    patches = os.listdir(slide_path)
    patches = [patch for patch in patches if patch.startswith("patch") and patch.endswith(".pt")]
    return transform, slide_path_end, patches

def create_slide_directories(path_to_embedding_dir: str, slide_path_end: str):
    """ Create the target directories for the concatenated embeddings that have the name
        of the corresponding slide. 
    
    Args:
        path_to_embedding_dir (str): Path to the directory containing the concatenated embeddings of the top n attended regions.
        slide_path_end (str): End of the slide path.
    """
    if not os.path.exists(os.path.join(path_to_embedding_dir, slide_path_end)):
        print(f"Creating directory {path_to_embedding_dir}/{slide_path_end}")
        # raise error if directory already exists
        os.makedirs(os.path.join(path_to_embedding_dir, slide_path_end), exist_ok=False) 

def load_patch_and_attn(slide_path: str, patch: str, attn_name: str, 
                        transform: transforms.Compose, device: torch.device) -> tuple:
    """ Load the patch and the attention map, transform the patch and prepare attention for usage. 
    
    Args:
        slide_path (str): Path to the slide directory.
        patch (str): Name of the patch.
        attn_name (str): Name of the attention map.
        transform (transforms.Compose): Transform for the patch.
        device (torch.device): Device on which the model is loaded.

    Returns:
        tuple: Tuple containing the transformed image, the label, the resized attention map and the original size of the image.
    """
    image, label = torch.load(os.path.join(slide_path, patch))
    image = transform(image.div(255)).unsqueeze(0).to(device) # since model expects batch dimension
    original_size = image.size(-1) # skip color and batch dimension

    # load attention map for selecting top n patches
    attn = torch.load(os.path.join(slide_path, attn_name))
    attn = attn.mean(0)
    resized_attn = T.Resize((original_size, original_size), antialias=True)(attn.unsqueeze(0)).squeeze()
    return image, label, resized_attn, original_size


# Assume WSI resolution is 4 000 000 px/m = 4 px/um -> 1 um = 4 px
SMALL_PATCH_SIZE = 100 # in um
SMALL_PATCH_SIZE_IN_PIXELS = 100 * 4 
BIG_PATCH_SIZE = 200 # in um
BIG_PATCH_SIZE_IN_PIXELS = 200 * 4 
def get_max_regions_and_sample_patches(image: torch.Tensor, resized_attn: torch.Tensor, original_size: int, 
                                       nr_of_attended_regions: int, allow_overlap: bool) -> tuple:
    """ Get the top n attended regions and sample the corresponding patches. 
    
    Args:
        image (torch.Tensor): Image of the patch.
        resized_attn (torch.Tensor): Resized attention map.
        original_size (int): Original size of the image.
        nr_of_attended_regions (int): Number of maximal attention values to compare between each map.
        allow_overlap (bool): Allow overlapping regions in the top n attended regions.

    Returns:
        tuple: Tuple containing the smaller patches and the bigger patches.
    """
    if allow_overlap:
        top_indices = get_max_indices(resized_attn, nr_of_attended_regions, original_size)
    else:
        top_indices = get_max_indices_without_overlap(resized_attn, nr_of_attended_regions, SMALL_PATCH_SIZE_IN_PIXELS, original_size)

    smaller_patches = get_patches_at_indices(image, original_size, top_indices, SMALL_PATCH_SIZE_IN_PIXELS)
    bigger_patches = get_patches_at_indices(image, original_size, top_indices, BIG_PATCH_SIZE_IN_PIXELS)
    # downsize patches such that they have the same size
    base_len = smaller_patches[0].size(-1)
    bigger_patches = [T.Resize((base_len, base_len), antialias=True)(patch) for patch in bigger_patches]
    return smaller_patches, bigger_patches


def get_max_indices(attn: torch.Tensor, nr_of_attended_regions: int, original_size: int) -> np.ndarray:
    """ Get the indices of the n highest values in the attention map. 
    
    Args:
        attn (torch.Tensor): Attention map.
        nr_of_attended_regions (int): Number of maximal attention values to compare between each map.
        original_size (int): Original size of the image.

    Returns:
        np.ndarray: Array containing the indices of the n highest values in the attention map.
    """
    attn_flat = attn.flatten()
    top_inds_1d = torch.topk(attn_flat, nr_of_attended_regions).indices
    # 1d -> 2d: x= ind % width, y = ind // width BUT we have to reverse the indices since height is the first dimension
    top_inds_2d = [(index.item() // original_size, index.item() % original_size) for index in top_inds_1d]
    top_inds_2d = np.array(top_inds_2d)
    # convert the to original size using UNI patch size and the downscale factor
    top_inds_2d = top_inds_2d * 16 # patch size from UNI is 16x16
    return top_inds_2d.astype(int)

def get_max_indices_without_overlap(attn: torch.Tensor, nr_of_attended_regions: int, 
                                    patch_size_in_pixels: int, original_size: int) -> np.ndarray:
    """ Get the indices of the n highest values in the attention map such that 
        the resulting 200um regions or patches do not overlap. 

    Args:
        attn (torch.Tensor): Attention map.
        nr_of_attended_regions (int): Number of maximal attention values to compare between each map.
        patch_size_in_pixels (int): Size of the patches in pixels.
        original_size (int): Original size of the image.

    Returns:
        np.ndarray: Array containing the indices of the n highest values in the
                    attention map such that the resulting 200um regions or patches do not overlap.    
    """
    top_inds_2d = []
    half_width = patch_size_in_pixels // 2
    for _ in range(nr_of_attended_regions):
        attn_flat = attn.flatten()
        # 1d -> 2d: x= ind % width, y = ind // width BUT we have to reverse the indices 
        top_ind = torch.argmax(attn_flat)
        x = top_ind % original_size
        y = top_ind // original_size
        # set the region around the maximum to zero
        x_min = max(0, x - half_width)
        x_max = min(original_size, x + half_width)
        y_min = max(0, y - half_width)
        y_max = min(original_size, y + half_width)
        # note: height is the first dimension -> y is the first index
        attn[y_min:y_max, x_min:x_max] = 0
        top_inds_2d.append((y, x))
    
    # convert the to original size using UNI patch size and the downscale factor
    top_inds_2d = np.array(top_inds_2d)
    top_inds_2d = top_inds_2d * 16 # patch size from UNI is 16x16
    return top_inds_2d.astype(int)

def get_patches_at_indices(full_image: torch.Tensor, original_size: int, 
                           top_indices: list, patch_size_in_pixels: int) -> torch.Tensor:
    """ Sample patch around the top indices and push them inside the image if 
        they are outside of the original image.
    
    Args:
        full_image (torch.Tensor): Image of the patch.
        original_size (int): Original size of the image.
        top_indices (list): List containing the indices of the n highest values in the attention map.
        patch_size_in_pixels (int): Size of the patches in pixels.

    Returns:
        torch.Tensor: Tensor containing the patches around the
                      top indices that are pushed inside the image if they are outside.
    """ 
    half_width = patch_size_in_pixels // 2
    patches = []
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
        # sample the patch -> note: height comes first
        patches.append(full_image[..., y_min:y_max, x_min:x_max]) # (B, C, H, W)
    return patches

def generate_and_save_concatenated_embeddings(model: torch.nn.Module, smaller_patches: list, 
                                              bigger_patches: list, label: int, emb_name: str):
    """ Generate embeddings for each patch in different resolutions, concatenate 
        them and save the resulting embedding. 

    Args:
        model (torch.nn.Module): Model for generating embeddings.
        smaller_patches (list): List containing the smaller patches.
        bigger_patches (list): List containing the bigger patches.
        label (int): Label of the patch.
        emb_name (str): Name of the embedding.    
    """
    with torch.no_grad():
        # generate embeddings for each patch in different resolutions (remove batch dimension again)
        smaller_patches = [model(patch).squeeze().detach().cpu() for patch in smaller_patches]
        bigger_patches = [model(patch).squeeze().detach().cpu() for patch in bigger_patches]
        # concatenate embeddings
        concatenated_emb = torch.cat(smaller_patches + bigger_patches, dim=0)
    torch.save((concatenated_emb,label), emb_name)

################################################################################################################

def main(patch_size: str, dataset: str, starting_slide: int, nr_of_slides: int, 
         nr_of_attended_regions: int, allow_overlap: bool):
    """ Main function for generating the concatenated embeddings of the top n attended regions.

    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset.
        starting_slide (int): Index of the first slide to start with.
        nr_of_slides (int): Number of slides to process.
        nr_of_attended_regions (int): Number of maximal attention values to compare between each map.
        allow_overlap (bool): Allow overlapping regions in the top n attended regions.
    """
    # get important parameters and link to the annotations directory
    model, device, path_to_class_dirs, path_to_embedding_dir, path_to_attention_dir = general_setup(patch_size, dataset, nr_of_attended_regions, allow_overlap)
    path_to_slides = get_slide_paths(path_to_class_dirs, starting_slide, nr_of_slides)

    for slide_path in tqdm(path_to_slides, total=len(path_to_slides)):
        # prepare for iteration over patches
        transform, slide_path_end, patches = prepare_for_patch_iteration(slide_path, path_to_embedding_dir)

        # generate embeddings and attention maps for each patch
        for patch in tqdm(patches, total=len(patches), desc=f"Generating concatenated top embeddings for slide at {slide_path}"):
            # check if embeddings already exist
            emb_name = os.path.join(path_to_embedding_dir, slide_path_end, patch)
            attn_name = os.path.join(path_to_attention_dir, slide_path_end, patch)
            if os.path.exists(emb_name):
                continue

            # load image, label, attention map and transform image
            image, label, resized_attn, original_size = load_patch_and_attn(slide_path, patch, attn_name, transform, device)
            # get the top n attended regions and sample the corresponding patches
            smaller_patches, bigger_patches = get_max_regions_and_sample_patches(image, resized_attn, original_size, nr_of_attended_regions, allow_overlap)
            # generate and save concatenated embeddings
            generate_and_save_concatenated_embeddings(model, smaller_patches, bigger_patches, label, emb_name)
            
            


if __name__ == '__main__':
    args = parse_arguments()
    start = time.time()
    main(**args)    
    print(f"Finished in {time.time() - start:.2f} seconds.")
    
