import os
import torch
import pandas as pd
from torch.utils.data import Dataset



"""
docker run --shm-size=100gb --gpus all --name ftoelkes_run1 -it -u `id -u $USER` --rm -v /home/ftoelkes/preprocessed_data/train_data:/data --rm -v /home/ftoelkes/code/lymphoma:/mnt ftoelkes_lymphoma python3 -m resnet_ddp.patchesdataset --patch_size="1024um" --data_dir="embeddings_dir" --data_specifier="patches"
"""

def parse_arguments() -> dict:
    """ Parse the arguments for the script.
    
    Returns:
        dict: Dictionary containing the arguments.
    """
    parser = argparse.ArgumentParser(description='Create annotation files for training and evaluating on patch level level for cross validation.')
    parser.add_argument("--patch_size", type=str, default="1024um", help="Size of the patches. Default is 1024um.")
    parser.add_argument('--data_specifier', default='', type=str, help='String specifying the patches to be used, e.g top_5_patches (default: "patches")')    
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply", "munich" or "all_data". (default: kiel)')
    parser.add_argument("--data_dir", default="embeddings_dir", type=str, help="Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'. (default: embeddings_dir)")
    args = parser.parse_args()
    return dict(vars(args))


class PatchesDataset(Dataset):
    """Class to load the patches dataset."""

    def __init__(self, annotations_file: str, parrent_dir: str, data_dir: str, data_specifier: str):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            parrent_dir (string): Path to the parrent directory of the data directory, which contains the patches sampled from different datasets.
            data_dir (string): Name of the directory containing the data: "data_dir", "embeddings_dir" or "embeddings_dir_not_normalized".
            data_specifier (string): String specifying the patches to be used, e.g top_5_patches or patches.
        """
        self.patches_frame = pd.read_csv(annotations_file)
        self.parrent_dir = parrent_dir
        self.data_dir = data_dir
        self.data_specifier = data_specifier

    def __len__(self) -> int:
        """ Return the length of the dataset.
        
        Returns:
            int: Length of the dataset
        """
        return len(self.patches_frame)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """ Get the patch and its label at the given index.

        Args:
            idx (int): Index of the patch to be loaded.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the patch and its label.
        """
        # load the patch and its label; 0th column is the dataset name, 1st column is the file name
        dataset = self.patches_frame.iloc[idx, 0]
        patch_name = self.patches_frame.iloc[idx, 1]
        patch_path = os.path.join(self.parrent_dir, dataset, self.data_specifier, self.data_dir, patch_name) 
        patch_with_label = torch.load(patch_path) 
        patch = patch_with_label[0] 
        label = patch_with_label[1]
        return patch, label
    

def main(patch_size: str, dataset: str, data_dir: str, data_specifier: str):
    """ Test the custom dataset class and time the dataloading.
    
    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset.
        data_dir (str): Name of the directory containing the data.
        data_specifier (str): String specifying the patches to be used, e.g top_5_patches (default: "patches")
    """
    annotations_dir = f"/data/{patch_size}/annotations/{dataset}/inner_5_fold_cv_patchesPercent_1.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0"
    train_csv = os.path.join(annotations_dir, "train_0.csv")

    dataset = PatchesDataset(train_csv, f"/data/{patch_size}", data_dir, data_specifier)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=1, prefetch_factor=10)
    
    model = resnet18(resnet_dim="2d", input_channels=3)
    model = model.to("cuda")

    print("len(dataset):", len(dataset))
    start = time.time()
    
    transform = T.Resize((512, 512))
    for batch_idx, (data, target) in enumerate(dataloader):
        
        if transform:
            data = transform(data)

        # send the batch to the gpu
        data = data.to("cuda")
        target = target.to("cuda")
        output = model(data.div(255))
        print("output.shape:", output.shape)

        if batch_idx % 10 == 0:
            print(f"batch_idx: {batch_idx}")
            print("data.shape:", data.shape)
            print("target.shape:", target.shape)
            break

    end = time.time()
    print(f"Time for dataloading {end-start}")

if __name__ == "__main__":
    import time
    import argparse

    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms.v2 as T 

    from resnet_ddp.resnet import resnet18
    args = parse_arguments()
    main(**args)
    
    
