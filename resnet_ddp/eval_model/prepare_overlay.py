import torch
import argparse
import os
import numpy as np

from tqdm import tqdm
from torch.nn import functional as F

from lymphoma.resnet_ddp.resnet import load_trained_model
from lymphoma.resnet_ddp import utils
from lymphoma.resnet_ddp.eval_model.test_best_model_ensemble import get_path_and_parameters
from lymphoma.resnet_ddp.eval_model.plot_overlay import setup_paths, get_coordinates_from_tiles_list, get_slide_dataloader

# dokcer command
"""
docker run --shm-size=400gb --gpus \"device=0\" --name ftoelkes_run01 -it -u `id -u $USER` --rm -v /sybig/home/ftoelkes/code:/mnt -v /sybig/home/ftoelkes/preprocessed_data/test_data:/data ftoelkes_lymphoma python3 -m lymphoma.resnet_ddp.eval_model.prepare_overlay --patch_size="1024um" --dataset="kiel" --eval_dataset="kiel" --data_specifier="patches" --data_dir=embeddings_dir --target_model="RESNET18,dd=embeddings_dir,e=50,lr=0.001,bs=256,wl=True,ls=0.1,p=10,s=42" 
"""

def parse_arguments() -> dict:
    """ Parse command line arguments. 
    
    Returns:
        dict: Dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the (augemented) test set')
    # test parameters
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply", "marr" or "all_data". (default: kiel)')
    parser.add_argument('-eds', '--eval_dataset', default='kiel', type=str, help='Name of the dataset to be used for the test data, note that its possible to eval the models on datasets they have not been trained on (default: kiel)')
    parser.add_argument('--data_specifier', default='patches', type=str, help='String specifying the patches to be used, e.g top_5_patches (default: "patches")')
    parser.add_argument("--data_dir", default="embeddings_dir", type=str, help="Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'. (default: embeddings_dir)")
    parser.add_argument('--slide_name', default='FL/80212-2018-0-HE-FL', type=str, help='Name of the slide to evaluate. (default: FL/80212-2018-0-HE-FL)')
    # other parameters
    parser.add_argument('--annotations_dir', default="inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0", type=str, help='Path to the directory where the different annotations files are located which is needed for the dataloader. (default: inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0)')
    parser.add_argument('--target_model', default="RESNET18,dd=embeddings_dir,e=50,lr=0.001,bs=256,wl=True,ls=0.1,p=10,s=42", type=str, help='Name of the target model. (default: RESNET18,dd=embeddings_dir,e=50,lr=0.001,bs=256,wl=True,ls=0.1,p=10,s=42)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    args = parser.parse_args()
    return dict(vars(args))


class EnsembleModel:
    """ Ensemble model class, that loads the models from a cross-validation run and predicts patches with them. """
    def __init__(self, base_path: str, nr_of_folds: int, resnet_type: str, resnet_dim: int, input_channels: int, gpu_id: int):
        self.base_path = base_path
        self.nr_of_folds = nr_of_folds
        self.resnet_type = resnet_type
        self.resnet_dim = resnet_dim    
        self.input_channels = input_channels
        self.gpu_id = gpu_id
        self.models = []

        print(f"Loading models from {base_path}")
        self.load_models()

    def load_models(self):
        """ Load the models from the base path. """
        for i in range(self.nr_of_folds):
            path = self.base_path + f"/best_model_fold_{i}.pt"
            print(f"Loading model from {path}")
            model = load_trained_model(self.resnet_type, self.resnet_dim, self.input_channels, path, self.gpu_id)
            model.eval()
            self.models.append(model)

    def predict(self, x) -> torch.Tensor:
        """ Get predictions for all models and average over them to get output of models as ensemble. 
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the ensemble model.
        """
        outputs = []
        for model in self.models:
            output = model(x)
            # softmax to get probabilities
            output = F.softmax(output, dim=1) # dim=1 because we have batch_size x classes
            outputs.append(output)
        return torch.stack(outputs).mean(dim=0) # mean over the ensemble models


def main(patch_size: str, dataset: str, eval_dataset: str, data_dir: str, data_specifier: str, 
         slide_name: str, annotations_dir: str, target_model: str, seed: int):
    """ Main function to prepare the overlay plot for a given slide.

    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset.
        eval_dataset (str): Name of the dataset to be used for the test data.
        data_dir (str): Name of the directory containing the data.
        data_specifier (str): String specifying the patches to be used.
        slide_name (str): Name of the slide to evaluate.
        annotations_dir (str): Path to the directory where the different annotations files are located.
        target_model (str): Name of the target model.
        seed (int): Random seed.
    """
    # get save parameters of best model, i.e. the one with the highest auc
    model_path = f"/mnt/lymphoma/resnet_ddp/experiments/{patch_size}/{dataset}/models_on_{data_specifier}/{annotations_dir}"
    if not os.path.exists(model_path) or model_path == "":
        resnet_dir, resnet_type = utils.get_best_model_dir(model_path, data_dir)
    else:
        resnet_dir = target_model
        resnet_type = resnet_dir.split(",")[0].lower()

    # set and update paths and parameters
    patch_transform, resnet_dim, input_channels = get_path_and_parameters(resnet_dir)
    output_dir = os.path.join(model_path, resnet_dir)

    # setup testing environment 
    gpu_id = "cuda:0" if torch.cuda.is_available() else "cpu"
    utils.general_setup(seed)

    # nr_of_folds = 1
    nr_of_folds = int(annotations_dir.split("_")[1])
    output_dir_dataset = os.path.join(output_dir, eval_dataset, slide_name.split("/")[1])
    os.makedirs(output_dir_dataset, exist_ok=True)

    # load ensemble model
    ensemble_model = EnsembleModel(output_dir, nr_of_folds, resnet_type, resnet_dim, input_channels, gpu_id)

    # load data
    path_to_data_dir = f"/data/{patch_size}/{dataset}/{data_specifier}/embeddings_dir"
    print(f"Path to image patches: {path_to_data_dir}")

    path_to_slide, patch_names, path_to_log = setup_paths(path_to_data_dir, slide_name, "embeddings_dir")

    # get infos important for plotting
    coordinates = get_coordinates_from_tiles_list(patch_names, tile_size_in_pixels=512)
    # get the dataloader for the slide
    embedding_dataloader = get_slide_dataloader(path_to_slide, patch_names)

    preds_and_coords = {}
    
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(embedding_dataloader), total=len(embedding_dataloader)):
            if patch_transform:
                x = patch_transform(x)
            x = x.to(gpu_id)
            output = ensemble_model.predict(x)
            _, pred = torch.max(output, 1)
            # save the prediction with the corresponding coordinate
            preds_and_coords[coordinates[i]] = pred.cpu().numpy()
    print("Finished predicting all patches.")

    # save the predictions
    save_path = os.path.join(output_dir_dataset, f"predictions_for_overlay_plot.npy")
    print(f"Saving predictions to {save_path}")
    np.save(save_path, preds_and_coords, allow_pickle=True)



if __name__ == "__main__":
    args = parse_arguments()
    main(**args)