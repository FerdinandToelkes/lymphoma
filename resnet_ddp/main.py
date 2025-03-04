import os
import argparse
import torch.distributed as dist
import time
import torch
import torchvision.transforms.v2 as T 
import pandas as pd
import numpy as np


from torch.utils.data import DataLoader 
from torch.utils.data.distributed import DistributedSampler # needed for distributed training

# imports from own code
import lymphoma.resnet_ddp.utils as utils
from lymphoma.resnet_ddp.patchesdataset import PatchesDataset
from lymphoma.resnet_ddp.resnet import load_train_objs
from lymphoma.resnet_ddp.trainer import Trainer

 
# cross validation with slide level validation
"""
screen -dmS ftoelkes_run0 sh -c '/home/ftoelkes/code/lymphoma/resnet_ddp/run_multiple_trainings.sh; exec bash'


screen -dmS train_resnet sh -c 'docker run --shm-size=400gb --gpus all --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /home/ftoelkes/code:/mnt -v /home/ftoelkes/preprocessed_data/train_data:/data ftoelkes_lymphoma torchrun --standalone --nproc_per_node=8 -m lymphoma.resnet_ddp.main --total_epochs=1 --save_every=1 --validate_every=1 --batch_size=256 --offset=0 --annotations_dir=inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0 --patch_size=1024um --data_dir=data_dir --dataset=kiel --data_specifier=patches -vm=slide --patience=10 -wl -lr=0.001 -ls=0.1 -wu=5 ; exec bash'
"""



def parse_arguments() -> dict:
    """ Parse command line arguments.
    
    Returns:
        dict: Dictionary containing all command line arguments.
    """
    parser = argparse.ArgumentParser(description='Resnet experiment.')
    # parameters needed to specify all the different paths
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('--data_specifier', default='patches', type=str, help='String specifying the patches to be used, e.g top_5_patches (default: "patches")')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply", "munich" or "all_data". (default: kiel)')
    parser.add_argument("--data_dir", default="embeddings_dir", type=str, help="Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'. (default: embeddings_dir)")
    # training parameters
    parser.add_argument('--resnet_type', default='resnet18', type=str, help='Type of ResNet to use. (default: resnet18)')
    parser.add_argument('-e', '--total_epochs', default=1, type=int, help='Total epochs to train the model. (default: 1)')
    parser.add_argument('-se', '--save_every', default=5, type=int, help='How often to save a snapshot. (default: 5)')
    parser.add_argument('-ve', '--validate_every', default=5, type=int, help='How often to validate the model. (default: 5)')
    parser.add_argument('-vm', '--validation_mode', default='slide', type=str, help='Whether to validate during training on patch or slide level. (default: slide)')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate for the optimizer. (default: 1e-3)')
    parser.add_argument('-wu', '--warmup_epochs', default=0, type=int, help='Number of warmup epochs for the learning rate scheduler. (default: 0)')
    parser.add_argument('-bs', '--batch_size', default=1000, type=int, help='Input batch size on each device (default: 256).')
    parser.add_argument('-wl', '--weighted_loss', default=False, action='store_true', help='Use weighted cross entropy loss for imbalanced datasets. (default: False, if flag is set: True)')
    parser.add_argument('-ls', '--label_smoothing', default=0, type=float, help='Use label smoothing for the cross entropy loss in range [0,1]. A typical value is 0.1 (default: 0, i.e. no label smoothing)')
    # other parameters
    parser.add_argument('--offset', default=0, type=int, help='Offset for GPU ids -> needed when running on specific gpus. (default: 1)')
    # parameters that will likely not be changed
    parser.add_argument('--annotations_dir', default="inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0", type=str, help='Name of the directory where the different annotations files are located which is needed for the dataloader. (default: inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0)')
    parser.add_argument('-p', '--patience', default=2, type=int, help='Patience for early stopping. (default: 2)')
    parser.add_argument('-s', '--seed', default=42, type=int, help='Random seed (default: 42)')
    return dict(vars(parser.parse_args())) 

def get_params_for_embs_or_patches(dataset: str, data_dir: str, annotations_dir: str, 
                                   batch_size: int, data_mount_dir: str, resize_target: int) -> dict:
    """ Setup for embeddings or patches.
    
    Args:
        dataset (str): Name of the dataset.
        data_dir (str): Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'.
        annotations_dir (str): Name of the directory where the different annotations files are located which is needed for the dataloader.
        batch_size (int): Batch size for training.
        data_mount_dir (str): Path to the mounted data directory.
        resize_target (int): Target size for resizing the patches.

    Returns:
        dict: Dictionary containing all parameters needed for training with embeddings or patches.
    """
    if "embeddings" in data_dir:
        print("Using UNI embeddings for training.")
        resnet_dim = "1d" # use 1d resnet for embeddings
        input_channels = 1
        patch_transform = torch.Tensor.float
    else:
        print("Using patches for training.")
        resnet_dim = "2d" # use 2d resnet for patches
        input_channels = 3
        # patch trafo: division by 255 and resize 
        patch_transform = T.Compose([utils.DivisionTransform(), 
                                     T.Resize((resize_target, resize_target), antialias=True)])
    label_transform = None
    path_to_annotations = os.path.join(data_mount_dir, "annotations", dataset, annotations_dir)
    # divide batch size by number of gpus, otherwise batch size would be too big: 
    # batch_size=32, world_size=4 -> batch_size would be 128!
    batch_size = batch_size // dist.get_world_size() 
    return {
        "resnet_dim": resnet_dim,
        "input_channels": input_channels,
        "patch_transform": patch_transform,
        "label_transform": label_transform,
        "path_to_annotations": path_to_annotations,
        "batch_size": batch_size
    }


def create_output_directory(output_dir: str):
    """ Creates output and plot directories if they do not exist. 
    
    Args:
        output_dir (str): Path to the output directory.

    Raises:
        ValueError: If the output directory already exists.
    """
    if not os.path.exists(output_dir):
        print(f"Creating output directory at {output_dir}")
        os.makedirs(output_dir)
    elif os.path.exists(output_dir):
        raise ValueError(f"Output directory already exists: {output_dir}")


def setup_for_cv_or_normal_training(annotations_dir: str, validation_mode: str) -> tuple[int, bool, dict]:
    """ Setup for cross validation or normal training. 
    
    Args:
        annotations_dir (str): Name of the directory where the different annotations files are located which is needed for the dataloader.
        validation_mode (str): Whether to validate during training on patch or slide level.

    Returns:
        tuple: Tuple containing the number of folds, whether to save the model and the statistics.
    """
    if "fold_cv" in annotations_dir:
        # assume name of form inner_5_fold_cv_patchesPercent_75.0_...
        nr_of_folds = int(annotations_dir.split("_")[1])
    else:
        nr_of_folds = 0

    save_model = True
    stats = {f"{validation_mode}_acc": [], f"{validation_mode}_loss": []}
    return nr_of_folds, save_model, stats


def get_data_loader(data_mount_dir: str, data_dir: str, path_to_annotations_dir: str, batch_size: int, 
                    mode: str, data_specifier: str, k: int = -1) -> DataLoader:
    """ Returns the dataloaders for training and validation. 
    
    Args:
        data_mount_dir (str): Path to the mounted data directory.
        data_dir (str): Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'.
        path_to_annotations_dir (str): Path to the annotations directory.
        batch_size (int): Batch size for training.
        mode (str): Whether to load training or validation data.
        data_specifier (str): String specifying the patches to be used, e.g top_5_patches or patches.
        k (int, optional): Fold number (default: -1)

    Returns:
        DataLoader: Dataloader for training or validation.
    """
    print("Setting up data and objects needed for training ...")
    if k >= 0:
        path_to_csv = os.path.join(path_to_annotations_dir, f"{mode.lower()}_{k}.csv")
    else:
        path_to_csv = os.path.join(path_to_annotations_dir, f"{mode.lower()}.csv")
    # create dataloader for training and validation, Note: apply any transforms later on the batches and not here since its faster
    print(f"Loading {mode.lower()} data from {path_to_csv} with data specifier {data_specifier} ...")
    data = PatchesDataset(path_to_csv, data_mount_dir, data_dir, data_specifier)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(data), 
                              drop_last=True, num_workers=1, prefetch_factor=10)
    # print total number of samples
    print(f"Total number of {mode.lower()} samples: {len(data)}")
    # check if batch size is bigger than the total number of validation samples
    if batch_size * dist.get_world_size() > len(data):
        print(f"Warning: The batch size must be smaller than the total number of {mode.lower()} samples devided by the number of gpus")
    return dataloader


def get_sorted_slide_df(path_to_annotations_dir: str, mode: str, k: int = -1) -> pd.DataFrame:
    """ Returns a dataframe with sorted slide names for specified mode for patch level annotation files. 
    
    Args:
        path_to_annotations_dir (str): Path to the annotations directory.
        mode (str): Whether to load training or validation data.
        k (int, optional): Fold number (default: -1)

    Returns:
        pd.DataFrame: Dataframe with sorted slide names.
    """
    if k >= 0:
        path_to_csv = os.path.join(path_to_annotations_dir, f"{mode.lower()}_{k}.csv")
    else:
        path_to_csv = os.path.join(path_to_annotations_dir, f"{mode.lower()}.csv")
    df = pd.read_csv(path_to_csv)
    # transform filename column to slide names
    df["filename"] = df["filename"].apply(lambda x: x.split("/")[0] + "/" + x.split("/")[1])
    # drop rows such that only unique slide names are left
    df_slides = df.drop_duplicates(subset=["filename"])
    df_slides = df_slides.sort_values(by="filename")
    return df_slides

def finalize_and_save_cv_stats(stats: dict, validation_mode: str, nr_of_folds: int, output_dir: str):
    """ Finalize cross validation statistics by adding the averages and their standard 
        errors and save them to a csv. 
        
    Args:
        stats (dict): Dictionary containing the model statistics.
        validation_mode (str): Whether to validate during training on patch or slide level.
        nr_of_folds (int): Number of folds used for cross validation.
        output_dir (str): Path to the output directory.
    """
    # average metrics over all folds and add to cv_stats
    stats["total_acc"] = np.mean(stats[f"{validation_mode}_acc"])
    stats["total_loss"] = np.mean(stats[f"{validation_mode}_loss"])
    # add standard errors of the mean e=std/sqrt(n) and add to cv_stats
    stats["std_err_acc"] = np.std(stats[f"{validation_mode}_acc"]) / np.sqrt(nr_of_folds)
    stats["std_err_loss"] = np.std(stats[f"{validation_mode}_loss"]) / np.sqrt(nr_of_folds)
    # switch to dataframe and save cv_stats
    path_to_cv_stats = os.path.join(output_dir, f"{nr_of_folds}_fold_cv_{validation_mode}_stats.csv")
    stats = pd.DataFrame(stats)
    stats.to_csv(path_to_cv_stats, index=False)
    print(f"Saved cv_stats:\n{stats}\n to {path_to_cv_stats}")

def save_training_parameters(save_every: int, validate_every: int, total_epochs: int, batch_size: int,
                            weighted_loss: bool, label_smoothing: float, output_dir: str, data_dir: str, 
                            path_to_annotations_dir: str):
    """ Saves the parameters used for training. 
    
    Args:
        save_every (int): How often to save a snapshot.
        validate_every (int): How often to validate the model.
        total_epochs (int): Total epochs to train the model.
        batch_size (int): Batch size for training.
        weighted_loss (bool): Whether to use weighted cross entropy loss for imbalanced datasets.
        label_smoothing (float): Use label smoothing for the cross entropy loss in range [0,1].
        output_dir (str): Path to the output directory.
        data_dir (str): Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'.
        path_to_annotations_dir (str): Path to the annotations directory
    """
    path = os.path.join(output_dir, 'training_parameters.txt')
    print("Saving training parameters at: ", path)
    with open(path, 'w') as f:   
        f.write(f"Parameters used for training:\n")
        f.write(f"save_every:       {save_every}\n")
        f.write(f"validate_every:   {validate_every}\n")
        f.write(f"total_epochs:     {total_epochs}\n")
        f.write(f"batch_size:       {batch_size}\n")
        f.write(f"weighted_loss:    {weighted_loss}\n")
        f.write(f"labels_smoothing: {label_smoothing}\n")
        f.write(f"\n")
        f.write(f"Additional information:\n")
        f.write(f"date:             {time.strftime('%d.%m.%y')}\n")
        f.write(f"data_dir:         {data_dir} (this is usually mounted, see below for actual directory used)\n")
        f.write(f"path_to_annotations_dir:  {path_to_annotations_dir}\n")

########################################################################################################################

# kwargs necessary for additional argument output_dir which is not in **args (see execution of main function)
def main(patch_size: str, dataset: str, data_dir: str, data_specifier: str, resnet_type: str, save_every: int, validate_every: int,    
         validation_mode: str, total_epochs: int, learning_rate: float, warmup_epochs: int, batch_size: int, weighted_loss: bool, 
         label_smoothing: float, offset: int, patience: int, seed: int, annotations_dir: str, output_dir: str, **kwargs):
    """ Main function for training a resnet model.

    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset.
        data_dir (str): Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'.
        data_specifier (str): String specifying the patches to be used, e.g top_5_patches or patches.
        resnet_type (str): Type of ResNet to use.
        save_every (int): How often to save a snapshot.
        validate_every (int): How often to validate the model.
        validation_mode (str): Whether to validate during training on patch or slide level.
        total_epochs (int): Total epochs to train the model.
        learning_rate (float): Learning rate for the optimizer.
        warmup_epochs (int): Number of warmup epochs for the learning rate scheduler.
        batch_size (int): Batch size for training.
        weighted_loss (bool): Use weighted cross entropy loss for imbalanced datasets.
        label_smoothing (float): Use label smoothing for the cross entropy loss in range [0,1].
        offset (int): Offset for GPU ids -> needed when running on specific gpus.
        patience (int): Patience for early stopping.
        seed (int): Random seed.
        annotations_dir (str): Name of the directory where the different annotations files are located which is needed for the dataloader.
        output_dir (str): Path to the output directory.
        **kwargs: Additional keyword arguments.
    """
    # setup things like seed, ddp or cuda
    print("Setting up ...")
    data_mount_dir = os.path.join("/data", patch_size)
    gpu_id = int(os.environ["LOCAL_RANK"]) + offset
    utils.general_setup(seed)
    utils.ddp_setup(gpu_id, seed)

    # get all parameters needed for training with embeddings or patches
    resize_target = 512 if "1024" in patch_size else 256
    setup_params = get_params_for_embs_or_patches(dataset, data_dir, annotations_dir, 
                                                  batch_size, data_mount_dir, resize_target)
    batch_size = setup_params['batch_size']
    
    # create output and plot directory if they do not exist
    if gpu_id == offset:
        create_output_directory(output_dir)
    dist.barrier() # synchronize all processes before starting training

    # assume name like inner_5_fold_cv_patchesPercent_1.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0
    tumor_probability_threshold = int(annotations_dir.split("tumorThreshold_")[-1].split("_")[0])

    # setup for cross validation or normal training
    nr_of_folds, save_model, stats = setup_for_cv_or_normal_training(annotations_dir, validation_mode)
    
    for k in range(nr_of_folds):
        dist.barrier() # synchronize all processes before starting next fold
        print(f"GPU {gpu_id} is training fold {k+1} of {nr_of_folds} ...")
        
        # get class weights for weighted cross entropy loss
        if weighted_loss:
            print("Calculating weights from training data for weighted cross entropy loss ...")
            weights_for_loss = utils.get_class_weights_from_train_data(setup_params['path_to_annotations'], gpu_id, k)
        else:
            print("No weights for loss are used.")
            weights_for_loss = None # default argument for weights in pytroch's cross entropy loss

        # setup model, optimizer and scheduler -> use own model defined in resnet.py
        model, optimizer, warmup_scheduler, main_scheduler = load_train_objs(resnet_type, setup_params['resnet_dim'], 
                                                      setup_params['input_channels'], gpu_id, 
                                                      learning_rate, warmup_epochs=warmup_epochs) 

        # create dataloader for training and validation (patch or slide level for validation)
        train_loader = get_data_loader(data_mount_dir, data_dir, setup_params['path_to_annotations'], 
                                       batch_size, "train", data_specifier, k)
        if validation_mode == "patch":
            patch_val_loader = get_data_loader(data_mount_dir, data_dir, setup_params['path_to_annotations'], 
                                               batch_size, "val", data_specifier, k)
            val_slide_df = None
        elif validation_mode == "slide":
            val_slide_df = get_sorted_slide_df(setup_params['path_to_annotations'], "val", k)
            patch_val_loader = None
        else:
            raise ValueError("Validation mode must be either 'patch' or 'slide'.")

        # setup trainer and start training
        statistics_path = f"training_statistics_{k}.csv"
        trainer = Trainer(gpu_id, model, train_loader, patch_val_loader, val_slide_df, setup_params['patch_transform'], 
                          setup_params['label_transform'], total_epochs, warmup_epochs, optimizer, warmup_scheduler, 
                          main_scheduler, save_every, save_model, weights_for_loss, label_smoothing, patience, offset, 
                          output_dir, data_mount_dir, data_dir, data_specifier, tumor_probability_threshold, statistics_path)
        print("Start training ...")
        # the returned statistics correspond to the saved model
        best_model_name = f"best_model_fold_{k}.pt" 
        results = trainer.train(validate_every, best_model_name)

        # statistics have been already averaged over gpus, so we can save them directly
        if gpu_id == offset:
            stats[f"{validation_mode}_acc"].append(results["best_acc"])
            stats[f"{validation_mode}_loss"].append(results["best_loss"])

    
    # compute final scores and save cv statistics
    if gpu_id == offset:
        finalize_and_save_cv_stats(stats, validation_mode, nr_of_folds, output_dir)
        

    # save the parameters used for training
    save_training_parameters(save_every, validate_every, total_epochs, batch_size * dist.get_world_size(), 
                             weighted_loss, label_smoothing, output_dir,
                             data_dir, setup_params['path_to_annotations'])


########################################################################################################################


def create_experiment_description(args, output_dir):
    """Creates experiment description, modifies some characters and uses the modified description as name for the output directory."""
    # create experiment description
    args['experiment_description'] = "{},dd={},e={},lr={},wu={},bs={},wl={},ls={},p={},s={}".format(args['resnet_type'].upper(), args['data_dir'], 
                                    args['total_epochs'], args['learning_rate'], args['warmup_epochs'], args['batch_size'], args['weighted_loss'],
                                    args['label_smoothing'], args['patience'], args['seed'])
    # replace some characters in experiment description
    args['experiment_description'] = args['experiment_description'].replace(" ", "")
    args['experiment_description'] = args['experiment_description'].replace("'", "")
    args['experiment_description'] = args['experiment_description'].replace("<", "")
    args['experiment_description'] = args['experiment_description'].replace(">", "")
    output_dir = "{}/{}".format(output_dir, args['experiment_description'])
    print(args['experiment_description'])
    return output_dir

if __name__ == "__main__":
    # parse command line arguments
    args = parse_arguments()

    # specify output directory
    output_dir = f"/mnt/lymphoma/resnet_ddp/experiments/{args['patch_size']}/{args['dataset']}/models_on_{args['data_specifier']}"
    output_dir = "{}/{}".format(output_dir, args['annotations_dir'])
    
    # create experiment description and modify some characters such that it can be used as name for the output directory
    output_dir = create_experiment_description(args, output_dir)

    # main function
    start = time.time()
    main(**args, output_dir=output_dir) 
    end = time.time()
    print(f"Training took {end - start:.2f} seconds")











