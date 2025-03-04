import os
import torch
import argparse
import time
import timm
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from huggingface_hub import login, hf_hub_download
from tqdm import tqdm

from .custom_vision_transformer import load_wrapper_model


# Use the run_generate_embeddings.sh script to run this script with the correct arguments


def parse_arguments() -> dict:
    """ Parse the arguments for the script. 
    
    Returns:
        dict: Dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Prepare normalization of patches.')
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply" or "munich". (default: kiel)')
    parser.add_argument('--target_size', default=2048, type=int, help='Target size to which the patches are resized to. (default: 2048)')
    parser.add_argument('--starting_slide', default=0, type=int, help='Index of the first slide to start with. (default: 0)')
    parser.add_argument('--nr_of_slides', default=1, type=int, help='Number of slides to process. (default: 1)')
    parser.add_argument('--no_normalize', default=False, action='store_true', help='Whether to normalize the patches. (default: False)')
    parser.add_argument('-sa', '--save_attentions', default=False, action='store_true', help='Whether to save the attention maps. (default: False)')
    args = parser.parse_args()
    return vars(args)

def load_uni_model(device: torch.device, 
                   local_dir: str = "/vit_large_patch16_256.dinov2.uni_mass100k") -> torch.nn.Module:
    """ Load the UNI model from the local directory. 
    
    Args:
        device (torch.device): Device on which the model should be loaded.
        local_dir (str): Local directory where the model is stored. (default: "/vit_large_patch16_256.dinov2.uni_mass100k")

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, 
                              num_classes=0, dynamic_img_size=True)
    model.load_state_dict(torch.load(os.path.join(local_dir, "model.pth"), map_location="cpu"), strict=True)
    model.to(device)
    model.eval()
    return model


def load_uni_model_from_website(local_dir: str = "/vit_large_patch16_256.dinov2.uni_mass100k"):
    """ Load the UNI model from the Hugging Face website.

    Args:
        local_dir (str): Local directory where the model is stored. (default: "/vit_large_patch16_256.dinov2.uni_mass100k")
    """
    login(add_to_git_credential=False)  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    # note vit_large_patch16_224 is one of the models that can be downloaded from the website
    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, 
                              num_classes=0, dynamic_img_size=True)

    # Load the state dict and save the model
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    torch.save(model.state_dict(), os.path.join(local_dir, "model.pth"))


def plot_attention(img: torch.Tensor, attention:  torch.Tensor, save_path: str):
    """ Plot the attention map on top of the image.

    Args:
        img (torch.Tensor): Image tensor.
        attention (torch.Tensor): Attention map tensor.
        save_path (str): Path where the image should be saved.
    """
    # see https://ai.plainenglish.io/visualizing-attention-in-vision-transformer-c871908d86de

    plt.figure(figsize=(10, 10))
    text = ["Original Image", "Head Mean"]
    save_path = save_path.replace(".pt", ".png")

    # note that attention and image are both tensors
    img = img.numpy().transpose(1, 2, 0)
    attention = attention.numpy()
    mean_attention = np.mean(attention, axis=0)

    for i, fig in enumerate([img, mean_attention]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(fig, cmap='inferno')
        if i == 1:
            plt.colorbar()
        plt.title(text[i])
        # write the shape of the attention map below the image
        plt.text(0, fig.shape[0] + 100, f"Shape: {mean_attention.shape}", fontsize=12)

    plt.savefig(os.path.join(save_path))
    plt.close()


def prepare_attention_maps(model: torch.nn.Module, image_size: int) -> torch.Tensor:
    """ Prepare the attention map from the last block of the model such that it can be visualized. 
    
    Args:
        model (torch.nn.Module): Model from which the attention map should be extracted.
        image_size (int): Size of the image for which the attention map should be extracted.

    Returns:
        torch.Tensor: Attention map tensor.
    """
    attn_map = model.blocks[-1].attn.attn_map.detach()

    # keep only the output patch attention
    nh = attn_map.shape[1]  # number of heads
    # drop batch dimension and cls token dimension that was kept for compatability 
    # see unsqueeze(2) in forward_wrapper
    attentions = attn_map[0, :, 0, :] 
    attentions = attentions.reshape(nh, -1)
    attentions = attentions.reshape(nh, image_size // 16, image_size // 16) # attention map has square shape
    attentions = attentions.cpu() #.numpy()
    return attentions

def make_class_directories(embedding_dir: str):
    """ Create the class directories in the embedding directory. 
    
    Args:
        embedding_dir (str): Path to the embedding directory.
    """
    os.mkdir(os.path.join(embedding_dir, "HL"))
    os.mkdir(os.path.join(embedding_dir, "DLBCL"))
    os.mkdir(os.path.join(embedding_dir, "CLL"))
    os.mkdir(os.path.join(embedding_dir, "FL"))
    os.mkdir(os.path.join(embedding_dir, "MCL"))
    os.mkdir(os.path.join(embedding_dir, "LTDS"))

def general_setup(patch_size: str, dataset: str, no_normalize: bool, save_attentions: bool) -> tuple:
    """ Setup the model, device, path to class directories and path to embedding and attention directory. 
    
    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset.
        no_normalize (bool): Whether to normalize the patches.
        save_attentions (bool): Whether to save the attention maps.

    Returns:
        tuple: Tuple containing the model, device, path to class directories
               and path to embedding and attention directory
    """
    torch.hub.set_dir('tmp/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if save_attentions:
        model = load_wrapper_model(device, local_dir="/model")
    else:
        model = load_uni_model(device, local_dir="/model")
        #load_uni_model_from_website(local_dir="./vit_large_patch16_256.dinov2.uni_mass100k")
    print("model loaded")
   
    path_to_class_dirs = os.path.join("/data", patch_size, dataset, "patches", "data_dir")
    path_to_embedding_dir, path_to_attention_dir = create_base_and_class_directories(patch_size, dataset, no_normalize, save_attentions)

    return model, device, path_to_class_dirs, path_to_embedding_dir, path_to_attention_dir

def create_base_and_class_directories(patch_size: str, dataset: str, 
                                      no_normalize: bool, save_attentions: bool) -> tuple:
    """ Create the base directories for the embeddings and attention maps and the class directories in which 
        the directories with the corresponding slide name will be created in. 
        
    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset.
        no_normalize (bool): Whether to normalize the patches.
        save_attentions (bool): Whether to save the attention maps.

    Returns:
        tuple: Tuple containing the path to the embedding directory and path to the attention directory.
    """
    # setup path to directory with different classes containing the slide directories with the patches
    path_to_embedding_dir = os.path.join("/data", patch_size, dataset, "patches", "embeddings_dir")
    path_to_attention_dir = os.path.join("/data", patch_size, dataset, "patches", "attentions_dir")

    if no_normalize:
        path_to_embedding_dir = path_to_embedding_dir + "_not_normalized"
        path_to_attention_dir = path_to_attention_dir + "_not_normalized"

    if not os.path.exists(path_to_embedding_dir):
        print(f"Creating directory {path_to_embedding_dir} and class directories")
        os.makedirs(path_to_embedding_dir, exist_ok=True) # create directory if it does not exist
        make_class_directories(path_to_embedding_dir)

    if save_attentions and not os.path.exists(path_to_attention_dir):
        print(f"Creating directory {path_to_attention_dir} and class directories")
        os.makedirs(path_to_attention_dir, exist_ok=True)
        make_class_directories(path_to_attention_dir)

    return path_to_embedding_dir, path_to_attention_dir

def get_slide_paths(path_to_class_dirs: str, starting_slide: int, nr_of_slides: int) -> list:
    """ Get the paths to the slides that should be processed.

    Args:
        path_to_class_dirs (str): Path to the class directories
        starting_slide (int): Index of the first slide to start with.
        nr_of_slides (int): Number of slides to process.

    Returns:
        list: List containing the paths to the slides that should be processed.
    """
    # get all slide paths from the different classes
    path_to_slides = []
    for c in os.listdir(path_to_class_dirs):
        class_dir = os.path.join(path_to_class_dirs, c)
        slides_in_class = os.listdir(class_dir)
        path_to_slides.extend([os.path.join(class_dir, slide) for slide in slides_in_class]) 
    path_to_slides.sort()
    print(f"Found {len(path_to_slides)} slides in {path_to_class_dirs}")
   
    check_input(starting_slide, nr_of_slides, path_to_slides)

    if nr_of_slides == -1:
        path_to_slides = path_to_slides[starting_slide:]
    else:
        path_to_slides = path_to_slides[starting_slide:starting_slide+nr_of_slides]
    return path_to_slides

def check_input(starting_slide: int, nr_of_slides: int, path_to_slides: list):
    """ Check if the input is valid.

    Args:
        starting_slide (int): Index of the first slide to start with.
        nr_of_slides (int): Number of slides to process.
        path_to_slides (list): List containing the paths to the slides that should be processed.

    Raises:
        ValueError: If the starting slide index or number of slides is greater than the number of slides available.
    """
    if starting_slide >= len(path_to_slides):
        raise ValueError(f"Starting slide index or number of slides is greater than the number of slides available. "
                         f"Only {len(path_to_slides)} slides available.")

    if nr_of_slides + starting_slide > len(path_to_slides):
        print(f"Number of slides to process is greater than the number of slides available. "
              f"Only {len(path_to_slides)} slides available. Processing all possible slides.")
        nr_of_slides = len(path_to_slides) - starting_slide

def create_slide_directories(path_to_embedding_dir: str, path_to_attention_dir: str, 
                             slide_path_end: str, save_attentions: bool):
    """ Create the target directories for the embeddings and attention maps 
        that have the name of the corresponding slide. 
    
    Args:
        path_to_embedding_dir (str): Path to the embedding directory.
        path_to_attention_dir (str): Path to the attention directory.
        slide_path_end (str): Name of the slide.
        save_attentions (bool): Whether to save the attention maps.
    """
    if not os.path.exists(os.path.join(path_to_embedding_dir, slide_path_end)):
        print(f"Creating directory {path_to_embedding_dir}/{slide_path_end}")
        os.makedirs(os.path.join(path_to_embedding_dir, slide_path_end), exist_ok=False) # raise error if directory already exists

    if save_attentions and not os.path.exists(os.path.join(path_to_attention_dir, slide_path_end)):
        print(f"Creating directory {path_to_attention_dir}/{slide_path_end}")
        os.makedirs(os.path.join(path_to_attention_dir, slide_path_end), exist_ok=False)

def main(patch_size: str, dataset: str, starting_slide: int, nr_of_slides: int, 
         target_size: int, no_normalize: bool, save_attentions: bool):
    """ Main function to generate the embeddings and attention maps for the patches.

    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset.
        starting_slide (int): Index of the first slide to start with.
        nr_of_slides (int): Number of slides to process.
        target_size (int): Target size to which the patches are resized to.
        no_normalize (bool): Whether to normalize the patches.
        save_attentions (bool): Whether to save the attention maps.
    """
    model, device, path_to_class_dirs, path_to_embedding_dir, path_to_attention_dir = general_setup(patch_size, dataset, no_normalize, save_attentions)

    path_to_slides = get_slide_paths(path_to_class_dirs, starting_slide, nr_of_slides)

    for slide_path in tqdm(path_to_slides, total=len(path_to_slides)):
        print(f"Processing slide {slide_path}")

        # we only need last two directories of the slide path
        # slide path is of the form /data/400um_patches/data_dir/FL/80212-2018-0-HE-FL
        slide_path_end = os.path.join(*slide_path.split("/")[-2:]) # * unpacks the list
        create_slide_directories(path_to_embedding_dir, path_to_attention_dir, slide_path_end, save_attentions)

        if no_normalize:
            transform = transforms.Compose([
                transforms.Resize((target_size, target_size), antialias=True),
            ])
        else:
            mean, std = torch.load(os.path.join(slide_path, "mean_std.pt"))
            transform = transforms.Compose([            
                transforms.Normalize(mean, std),
                transforms.Resize((target_size, target_size), antialias=True),
                # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        # get all patches in the slide directory
        patches = os.listdir(slide_path)
        patches = [patch for patch in patches if patch.startswith("patch") and patch.endswith(".pt")]

        # generate embeddings and attention maps for each patch
        for patch in tqdm(patches, total=len(patches), desc=f"Generating embeddings for slide at {slide_path}"):
            emb_name = os.path.join(path_to_embedding_dir, slide_path_end, patch)
            attn_name = os.path.join(path_to_attention_dir, slide_path_end, patch)
            if os.path.exists(emb_name) and os.path.exists(attn_name):
                continue
            image, label = torch.load(os.path.join(slide_path, patch))
            image = transform(image.div(255)).unsqueeze(0).to(device) # since model expects batch dimension
            with torch.no_grad():
                embedding = model(image).squeeze().detach().cpu() # remove batch dimension again
            torch.save((embedding,label), emb_name)

            if save_attentions:
                image_size = image.shape[-1]
                attentions = prepare_attention_maps(model, image_size)    
                torch.save(attentions, attn_name)
            
            


if __name__ == '__main__':
    args = parse_arguments()
    start = time.time()
    main(**args)    
    print(f"Finished in {time.time() - start:.2f} seconds.")
    