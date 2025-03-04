import os
import torch
import pandas as pd                                                   # needed for creating annotations file
import torch.backends.cudnn as cudnn                                  # needed for cudnn auto-tuner
import torchvision.transforms.v2 as T                                 # needed for transforms


from torch import distributed as dist                                 # needed for distributed training
from torch.distributed import init_process_group                      # needed for distributed training                         


########################################################################################################################################################

def ddp_setup(gpu_id: int, seed: int):
    """
    Setup for distributed data parallel training.

    Args:
        gpu_id (int): Id of the GPU to use
        seed (int): Seed for all GPUs
    """
    init_process_group(backend="nccl")
    torch.cuda.set_device(gpu_id)
    # set seed for all gpus
    torch.cuda.manual_seed_all(seed)

def general_setup(seed: int, benchmark=True, hub_dir: str ='/mnt/'):
    """ General setup for training. 
    
    Args:
        seed (int): Seed for all GPUs
        benchmark (bool): Enable cudnn auto-tuner to find best algorithm for current hardware (default: True)
        hub_dir (str, optional): Directory for torch.hub -> where to save downloaded models (default: '/mnt/')
    """
    # en- or disable cudnn auto-tuner to find best algorithm for current hardware
    cudnn.benchmark = benchmark
    # set directory for torch.hub -> where to save downloaded models
    torch.hub.set_dir(hub_dir)
    torch.manual_seed(seed)

        
########################################################################################################################################################

def get_best_model_dir(path_to_experiments: str, data_dir: str) -> tuple[str, str]:
    """ Get the directory of the best model based on the highest auc from the hyperparameter search 
    
    Args:
        path_to_experiments (str): Path to the experiments directory
        data_dir (str): Directory of the data, i.e. "embeddings_dir", "data_dir" or "embeddings_not_normalized_dir"
    
    Returns:
        tuple[str, str]: Directory of the best model and the type of the resnet used
    """
    path_to_summary_csv = os.path.join(path_to_experiments, "models_summary.csv")
    if not os.path.exists(path_to_summary_csv):
        raise ValueError(f"Summary csv file does not exist: {path_to_summary_csv}\nPlease run summarize_cv_results.py first.")
    df = pd.read_csv(path_to_summary_csv)
    # backwards compatibility if ue=True or False is in the model name rename from
    # RESNET18,e=50,lr=0.0001,bs=256,ue=True,wl=True,ls=0.1,p=5,s=42 to RESNET18,dd=embeddings_dir,e=50,lr=0.0001,bs=256,wl=True,ls=0.1,p=5,s=42
    df["model_name"] = df["model_name"].str.replace("ue=True", "dd=embeddings_dir")
    df["model_name"] = df["model_name"].str.replace("ue=False", "dd=data_dir")
    # only consider models with data_dir in the name
    df = df[df["model_name"].str.contains(data_dir)]
    resnet_dir = df.loc[df['mean_slide_auc'].idxmax()]  
    resnet_dir = resnet_dir["model_name"]
    resnet_type = resnet_dir.split(",")[0].lower()
    return resnet_dir, resnet_type

########################################################################################################################################################

def get_class_weights_from_train_data(path_to_annotations: str, gpu_id: int, k: int = -1, 
                                      num_classes: int = 7) -> torch.Tensor:
    """ Compute class weights for loss from train data on patch level. 
    
    Args:
        path_to_annotations (str): Path to the annotations directory
        gpu_id (int): Id of the GPU
        k (int): Fold number (default: -1)
        num_classes (int): Number of classes (default: 7)

    Returns:
        torch.Tensor: Class weights for weighted cross entropy loss
    """
    if k >= 0:
        path_to_train_csv = os.path.join(path_to_annotations, f"train_{k}.csv")
    else:
        path_to_train_csv = os.path.join(path_to_annotations, "train.csv")
    train_df = pd.read_csv(path_to_train_csv)
    weights_for_loss = torch.ones(num_classes)
    # compute weights as done in https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75
    for i in range(num_classes):
        divisor = (train_df["label"] == i).sum()
        weights_for_loss[i] = len(train_df) / divisor if divisor != 0 else 1 # if divisor is 0, set weight to 1
    print(f"Class weights for loss: {weights_for_loss}")
    return weights_for_loss.to(gpu_id)


########################################################################################################################################################

def prepare_batch(x: torch.Tensor, y: torch.Tensor, patch_transform: T.Compose, 
                  label_transform: T.Compose, gpu_id: int) -> tuple:
    """ Prepare a batch for training or testing. 
    
    Args:
        x (torch.Tensor): Input data
        y (torch.Tensor): Labels
        patch_transform (T.Compose): Transformations for the input data
        label_transform (T.Compose): Transformations for the labels
        gpu_id (int): Id of the GPU

    Returns:
        tuple: Transformed input data
    """
    # apply patch and label transforms
    if patch_transform:
        x = patch_transform(x)
    if label_transform:
        y = label_transform(y)
    # move patches and labels to gpu
    x = x.to(gpu_id)
    y = y.to(gpu_id)
    return x, y

def calculate_and_append_statistics(loss_tensor: torch.Tensor, corrects_tensor: torch.Tensor, data_len: torch.Tensor, statistics_key_loss: str, statistics_key_acc: str, epoch: int, mode: str, gpu_id: int, statistics: dict) -> torch.Tensor:
    """Calculates and saves statistics (loss and accuracy) averaging over all GPUs.
    
    Args:
        loss_tensor (torch.Tensor): Tensor containing the loss
        corrects_tensor (torch.Tensor): Tensor containing the number of correct predictions
        data_len (torch.Tensor): Tensor containing the length of the data
        statistics_key_loss (str): Key to save the loss in the statistics dictionary
        statistics_key_acc (str): Key to save the accuracy in the statistics dictionary
        epoch (int): Current epoch
        mode (str): Training, validation or testing
        gpu_id (int): Id of the GPU
        statistics (dict): Dictionary to save the statistics

    Returns:
        torch.Tensor: The loss tensor
    """
    # Average over all GPUs
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(corrects_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(data_len, op=dist.ReduceOp.SUM)
    
    # Calculate loss and accuracy
    epoch_loss = loss_tensor / data_len
    epoch_acc = corrects_tensor.double() / data_len
    
    # Append statistics
    statistics[statistics_key_loss].append(epoch_loss.cpu().item())
    statistics[statistics_key_acc].append(epoch_acc.cpu().item())
    
    # Print results
    print(f"[GPU{gpu_id}] Epoch {epoch} | {mode} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return epoch_loss

class DivisionTransform:
    """ Custom patch transform for division by 255 such that it can be directly combined with normalization. """
    def __call__(self, img: torch.Tensor, dividor: int = 255) -> torch.Tensor:
        """ Divide the image by the dividor.

        Args:
            img (torch.Tensor): Image to divide
            dividor (int): Dividor (default: 255)

        Returns:
            torch.Tensor: Image divided by the dividor
        """
        return img.div(dividor)


