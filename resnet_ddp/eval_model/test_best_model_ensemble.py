import os
import time
import torch
import argparse
import pandas as pd
import torchvision.transforms.v2 as T 
import numpy as np


# import own modules
import lymphoma.resnet_ddp.utils as utils
from lymphoma.resnet_ddp.resnet import load_trained_model
from .slide_tester import Tester

NR_OF_CLASSES = 7

# docker command to run this script for models based on patches
"""
screen -dmS test_best_model_ensemble sh -c 'docker run --shm-size=400gb --gpus all --name ftoelkes_run01 -it -u `id -u $USER` --rm -v /sybig/home/ftoelkes/code:/mnt -v /sybig/home/ftoelkes/preprocessed_data/train_data:/train_data -v /sybig/home/ftoelkes/preprocessed_data/test_data:/data ftoelkes_lymphoma torchrun --standalone --nproc_per_node=8 -m lymphoma.resnet_ddp.eval_model.test_best_model_ensemble --offset=0 --patch_size="1024um" --dataset="marr" --eval_dataset="kiel" --data_specifier="patches" --data_dir=embeddings_dir --target_model="RESNET18,dd=embeddings_dir,e=50,lr=0.001,bs=256,wl=True,ls=0.1,p=10,s=42" --excluded_class=HL; exec bash'

screen -dmS test_best_model_ensemble sh -c 'docker run --shm-size=400gb --gpus all --name ftoelkes_run01 -it -u `id -u $USER` --rm -v /sybig/home/ftoelkes/code:/mnt -v /sybig/home/ftoelkes/preprocessed_data/train_data:/train_data -v /sybig/home/ftoelkes/preprocessed_data/test_data:/data ftoelkes_lymphoma torchrun --standalone --nproc_per_node=8 -m lymphoma.resnet_ddp.eval_model.test_best_model_ensemble --offset=0 --patch_size="1024um" --dataset="kiel" --eval_dataset="marr" --data_specifier="patches" --annotations_dir="inner_5_fold_cv_patchesPercent_1.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0" ; exec bash'
"""


def parse_arguments() -> dict:
    """ Parse the arguments for the test script. 
    
    Returns:
        dict: Dictionary with the arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the (augemented) test set')
    # test parameters
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply", "marr" or "all_data". (default: kiel)')
    parser.add_argument('-eds', '--eval_dataset', default='kiel', type=str, help='Name of the dataset to be used for the test data, note that its possible to eval the models on datasets they have not been trained on (default: kiel)')
    parser.add_argument('--excluded_class', default='', type=str, help='Class to be excluded from the evaluation. (default: "")')
    parser.add_argument('--data_specifier', default='patches', type=str, help='String specifying the patches to be used, e.g top_5_patches (default: "patches")')
    parser.add_argument("--data_dir", default="embeddings_dir", type=str, help="Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'. (default: embeddings_dir)")
    parser.add_argument('--offset', default=0, type=int, help='Offset for GPU ids -> needed when running on specific gpus')
    # other parameters
    parser.add_argument('--annotations_dir', default="inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0", type=str, help='Path to the directory where the different annotations files are located which is needed for the dataloader. (default: inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0)')
    parser.add_argument('--target_model', default="RESNET18,dd=embeddings_dir,e=50,lr=0.001,bs=256,wl=True,ls=0.1,p=10,s=42", type=str, help='Name of the target model. (default: RESNET18,dd=embeddings_dir,e=50,lr=0.001,bs=256,wl=True,ls=0.1,p=10,s=42)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    args = parser.parse_args()
    return dict(vars(args))

def get_path_and_parameters(resnet_dir: str) -> tuple:
    """ Get the path to the resnet model and the parameters for the patch transformation.

    Args:
        resnet_dir (str): Path to the resnet model.

    Returns:
        tuple: Patch transformation and resnet dimension
    """
    if "embeddings" in resnet_dir: # embeddings were used during training
        print("Using embeddings for testing")
        patch_transform = torch.Tensor.float
        resnet_dim = "1d"
        input_channels = 1
    else:
        print("Using patches for testing")
        patch_transform = T.Compose([utils.DivisionTransform(), 
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        resnet_dim = "2d"
        input_channels = 3
    return patch_transform, resnet_dim, input_channels


def update_results_of_fold(results: dict, probs_of_all_folds: np.ndarray, level: str) -> tuple:
    """ Get the results of the current fold and save them to npz files and update predictions for fold ensemble. 
    
    Args:
        results (dict): Dictionary with the results of the current fold.
        probs_of_all_folds (np.ndarray): Array with the predictions of all folds.
        level (str): Level of the predictions, i.e. "slides" or "patches".

    Returns:
        tuple: Array with the predictions of all folds, the ground truth labels and the predictions of the current fold.
    """
    # get the probabilities and labels of the current fold
    probs = results[f"probs_and_labels_for_all_{level}"]["probs"]
    probs = np.array(probs).reshape(-1, NR_OF_CLASSES) # ensure correct shape
    labels = results[f"probs_and_labels_for_all_{level}"]["labels"]
    
    # add the probabilities of the current fold to the array of all folds
    len1 = len(probs_of_all_folds)
    probs_of_all_folds += probs
    len2 = len(probs_of_all_folds)
    if len1 != len2:
        raise ValueError(f"Length of probs_of_all_folds changed from {len1} to {len2}")
    return probs_of_all_folds, labels, probs

def save_results_of_fold(probs: np.ndarray, labels: np.ndarray, k: int, 
                         output_dir_dataset: str, level: str, excluded_class: str):
    """ Save the results of the current fold to npz files.

    Args:
        probs (np.ndarray): Array with the predictions of the current fold.
        labels (np.ndarray): Array with the ground truth labels of the current fold.
        k (int): Number of the current fold.
        output_dir_dataset (str): Directory where the results should be saved.
        level (str): Level of the predictions, i.e. "slide" or "patch".
    """
    # save the probabilities and labels of current fold
    if excluded_class != "":
        level = f"{level}_without_{excluded_class}"
    path_to_probs_and_labels = os.path.join(output_dir_dataset, f"{level}_preds_labels_fold_{k}.npz")
    np.savez(path_to_probs_and_labels, probs=probs, labels=labels)


def get_results_of_fold(results: dict, level: str) -> tuple:
    """ Get the results of the current fold and convert probabilities to numpy array.
    
    Args:
        results (dict): Dictionary with the results of the current fold.

    Returns:
        tuple: Array with the predictions of all folds and the ground truth labels.
    """
    probs_of_all_folds = results[f"probs_and_labels_for_all_{level}"]["probs"]
    probs_of_all_folds = np.array(probs_of_all_folds).reshape(-1, NR_OF_CLASSES)
    labels = results[f"probs_and_labels_for_all_{level}"]["labels"]
    return probs_of_all_folds, labels

def save_ensemble_predictions(probs_of_all_folds: np.ndarray, labels: np.ndarray, nr_of_folds: int, 
                              output_dir_dataset: str, level: str, excluded_class: str):
    """ Save the ensemble predictions and ground truth labels.

    Args:
        probs_of_all_folds (np.ndarray): Array with the predictions of all folds.
        labels (np.ndarray): Array with the ground truth labels.
        nr_of_folds (int): Number of folds.
        output_dir_dataset (str): Directory where the results should be saved.
        level (str): Level of the predictions, i.e. "slide" or "patch".
    """
    # average the probabilities of all folds and compute the auc
    probs_of_all_folds /= nr_of_folds
    probs_and_labels = {"probs": probs_of_all_folds, "labels": labels}
    if excluded_class != "":
        level = f"{level}_without_{excluded_class}"
    # save the probabilities and labels for all slides
    path_to_probs_and_labels = os.path.join(output_dir_dataset, f"{level}_ensemble_predictions_and_ground_truth_labels.npz")
    np.savez(path_to_probs_and_labels, **probs_and_labels)
 
   
def main(patch_size: str, dataset: str, eval_dataset: str, excluded_class: str, data_dir: str, 
         data_specifier: str, offset: int, annotations_dir: str, target_model: str, seed: int):
    """ Main function to evaluate the best model on the test set.

    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset.
        eval_dataset (str): Name of the dataset to be used for the test data.
        excluded_class (str): Class to be excluded from the evaluation.
        data_dir (str): Name of the directory containing the data.
        data_specifier (str): String specifying the patches to be used.
        offset (int): Offset for GPU ids.
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

    # read tumor probability threshold from the name of the annotations directory
    tumor_probability_threshold = int(annotations_dir.split("tumorThreshold_")[1].split("_")[0])
    if tumor_probability_threshold > 100 or tumor_probability_threshold < 0:
        raise ValueError(f"Tumor probability threshold must be between 0 and 100 but is {tumor_probability_threshold}")

    # setup testing environment 
    gpu_id = int(os.environ["LOCAL_RANK"]) + offset
    utils.general_setup(seed)
    utils.ddp_setup(gpu_id, seed)


    # get slide names with datasets as df 
    if eval_dataset == dataset:
        path_to_slides_split = os.path.join("/train_data", patch_size, "annotations", dataset, "train_test_slides_split.csv")
        df = pd.read_csv(path_to_slides_split)
        test_slide_df = df[df[f"test"] == 1]
        
    # get all slide names from the eval dataset and put them in a df  
    else:
        path_to_class_dirs = os.path.join("/data", patch_size, eval_dataset, data_specifier, data_dir)
        class_dirs = os.listdir(path_to_class_dirs)
        class_dirs = [c for c in class_dirs if c != excluded_class]
        test_slides = []
        for c in class_dirs:
            slides = os.listdir(os.path.join(path_to_class_dirs, c))
            slides = [os.path.join(c,s) for s in slides if os.path.isdir(os.path.join(path_to_class_dirs, c, s))]
            test_slides += slides
            
        test_slide_df = pd.DataFrame(test_slides, columns=["filename"])
        test_slide_df["dataset"] = eval_dataset
        
        

    nr_of_folds = int(annotations_dir.split("_")[1])
    output_dir_dataset = os.path.join(output_dir, eval_dataset)
    os.makedirs(output_dir_dataset, exist_ok=True)
    for k in range(nr_of_folds):
        # load model for evaluation
        path_to_model = os.path.join(output_dir, f"best_model_fold_{k}.pt")
        print(f"Loading model from: {path_to_model}")
        model = load_trained_model(resnet_type, resnet_dim, input_channels, path_to_model, gpu_id)

        # initialize an tester instance and start evaluating
        label_transform = None
        tester = Tester(gpu_id, model, test_slide_df, patch_transform, label_transform, 
                        tumor_probability_threshold, offset, output_dir, data_specifier)
        print("Start testing ...")
        data_mount_dir = os.path.join("/data", patch_size)
        results = tester.run_testing(data_mount_dir, data_dir, data_specifier)

        # save the results if we are on root gpu
        if gpu_id == offset:
            #tester.save_wrong_classified_slides()
            if k > 0 and slide_labels != results["probs_and_labels_for_all_slides"]["labels"]: 
                raise ValueError("Ground truth labels are not the same for all folds")
            
            if k == 0:
                # since we need the "if" anyway, we can initialize the array for the slide probs here
                slide_probs_of_all_folds, slide_labels = get_results_of_fold(results, "slides")
                # since we do not know the number of patches beforehand, we need to initialize the array for the patch probs
                patch_probs_of_all_folds =  results["probs_and_labels_for_all_patches"]["probs"]
                patch_probs_of_all_folds = np.array(patch_probs_of_all_folds).reshape(-1, NR_OF_CLASSES)
                patch_labels = results["probs_and_labels_for_all_patches"]["labels"]

                # save the results of the first fold
                save_results_of_fold(slide_probs_of_all_folds, slide_labels, k, output_dir_dataset, "slide", excluded_class)
                save_results_of_fold(patch_probs_of_all_folds, patch_labels, k, output_dir_dataset, "patch", excluded_class)

            else:
                # append the probabilities of the current fold to the array of all folds
                slide_probs_of_all_folds, slide_labels, slide_probs = update_results_of_fold(results, slide_probs_of_all_folds, "slides")
                patch_probs_of_all_folds, patch_labels, patch_probs = update_results_of_fold(results, patch_probs_of_all_folds, "patches")

                # save the results of the current fold
                save_results_of_fold(slide_probs, slide_labels, k, output_dir_dataset, "slide", excluded_class)
                save_results_of_fold(patch_probs, patch_labels, k, output_dir_dataset, "patch", excluded_class)
            
            


    # final predictions are the average of the predictions of all folds
    if gpu_id == offset:
        save_ensemble_predictions(slide_probs_of_all_folds, slide_labels, nr_of_folds, output_dir_dataset, "slide", excluded_class)
        save_ensemble_predictions(patch_probs_of_all_folds, patch_labels, nr_of_folds, output_dir_dataset, "patch", excluded_class)
    
    
    


if __name__ == "__main__":
    args = parse_arguments()
    start = time.time()
    main(**args)
    end = time.time()
    print(f"Testing took {end-start:.2f}s")


