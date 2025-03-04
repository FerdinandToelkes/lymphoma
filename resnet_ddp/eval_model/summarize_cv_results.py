import argparse
import os
import pandas as pd
import re


# from .patch_tester import add_total_row

"""
docker run --shm-size=400gb --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /sybig/home/ftoelkes/code/lymphoma:/mnt ftoelkes_lymphoma python3 -m resnet_ddp.eval_model.summarize_cv_results --patch_size="1024um" --data_specifier="patches" --dataset="kiel"  

docker run --shm-size=400gb --gpus \"device=0\" --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /sybig/home/ftoelkes/code/lymphoma:/mnt ftoelkes_lymphoma python3 -m resnet_ddp.eval_model.summarize_cv_results --patch_size="1024um" --data_specifier="patches" --dataset="all_data"  --annotations_dir="inner_5_fold_cv_patchesPercent_1.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0"
"""


def parse_arguments() -> dict:
    """ Parse the arguments of the script.

    Returns:
        dict: Dictionary containing the arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the (augemented) test set')
    # parameters needed to specify the path to the results
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('--data_specifier', default='patches', type=str, help='String specifying the patches to be used, e.g top_5_patches (default: "patches")')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply", "marr" or "all_data". (default: kiel)')
    parser.add_argument('--resnet_type', default='resnet18', type=str, help='Type of ResNet to use. (default: resnet18)')
    # other parameters
    parser.add_argument('--annotations_dir', default="inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0", type=str, help='Path to the directory where the different annotations files are located which is needed for the dataloader. (default: inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0)')
    parser.add_argument('--eval_type', default='slide', type=str, help='Type of evaluation used during training. Either "patch" or "slide". (default: slide)')
    args = parser.parse_args()
    return dict(vars(args))


def get_models_to_evaluate(path_to_results: str, resnet_type: str) -> list:
    """ Get the models to evaluate from the results directory ordered by learning rate and label smoothing parameter.

    Args:
        path_to_results (str): Path to the directory where the results are saved
        resnet_type (str): Type of ResNet to use

    Returns:
        list: List of models to evaluate
    """
    models_to_evaluate = os.listdir(path_to_results)
    models_to_evaluate = [m for m in models_to_evaluate if m.startswith(f"{resnet_type.upper()}")]
    # sort after lr AND ls in name of form RESNET18,e=75,lr=0.0001,bs=256,ue=True,...
    # such that the models with the highest learning rate and lowest label smoothing are first
    models_to_evaluate.sort(key=lambda x: 1/float(re.findall(r"lr=(\d+\.\d+)", x)[0]) + float(re.findall(r"ls=(\d+\.\d+)", x)[0])) 
    return models_to_evaluate

def load_model_stats_and_append_to_dict(stats_from_all_models: dict, model: str, path_to_results: str, 
                                        stats_file_name: str, eval_type: str) -> None:
    """ Load the statistics of a model and append them to a dictionary.

    Args:
        stats_from_all_models (dict): Dictionary containing the statistics of all models
        model (str): Name of the model
        path_to_results (str): Path to the directory where the results are saved
        stats_file_name (str): Name of the csv file containing the statistics
        eval_type (str): Type of evaluation used during training. Either "patch" or "slide"
    """
    base_path = os.path.join(path_to_results, model)

    # Load the statistics of cv if available
    stats_path = os.path.join(base_path, stats_file_name)
    if not os.path.exists(stats_path):
        print(f"Skipping {model} because {stats_path} does not exist.")
        return None
    else:
        mean_acc, mean_loss, std_error_acc, std_error_loss = get_metrics_from_cv_results(stats_path)

    # extract parameters from model name
    model_params = extract_params_from_model_name(model)

    # Append the results to the dictionary
    append_stats_to_dict(stats_from_all_models, model, model_params, mean_acc, mean_loss, std_error_acc, std_error_loss, eval_type)
    return None

def get_metrics_from_cv_results(stats_path: str) -> tuple:
    """ Get the mean accuracy, mean loss, standard error of accuracy and standard error of loss from the cv results. 
    
    Args:
        stats_path (str): Path to the csv file with the cv results

    Returns:
        tuple: mean_acc, mean_loss, std_error_acc, std_error_loss
    """
    df_stats = pd.read_csv(stats_path)
    # note that for the total acc etc. the whole columns contain the same value -> take the first one
    mean_acc = df_stats['total_acc'][0]
    mean_loss = df_stats['total_loss'][0]
    std_error_acc = df_stats['std_err_acc'][0]
    std_error_loss = df_stats['std_err_loss'][0]
    return mean_acc, mean_loss, std_error_acc, std_error_loss

def extract_params_from_model_name(model: str) -> str:
    """ Get the parameters of the model which were important for hyperparameter search as a string. 
        Here we filter for learning rate, weighted loss and label smoothing. 

    Args:
        model (str): Name of the model

    Returns:
        str: Parameters of the model 
    """
    # \d*: match zero or more digits, \d+: match one or more digits, ,s=\d+: match ",s=" followed by one or more digits
    pattern = r"(RESNET\d*|e=\d+|bs=\d+|p=\d+|,s=\d+)"
    # Remove everything that is not lr, wl, ls or wu
    modified_model_name = re.sub(pattern, "", model).strip(", ")
    # Remove any unnecessary commas
    model_params = re.sub(r",+", ",", modified_model_name).strip(", ")
    return model_params

def append_stats_to_dict(stats_from_all_models: dict, model: str, model_params: str, mean_acc: float, 
                         mean_loss: float, std_error_acc: float, std_error_loss: float, eval_type: str):
    """ Append the statistics of a model to a dictionary.

    Args:
        stats_from_all_models (dict): Dictionary containing the statistics of all models
        model (str): Name of the model
        model_params (str): Parameters of the model (lr, wl, ls)
        mean_acc (float): Mean accuracy of the model
        mean_loss (float): Mean loss of the model
        std_error_acc (float): Standard error of the accuracy
        std_error_loss (float): Standard error of the loss
        eval_type (str): Type of evaluation used during training. Either "patch" or "slide"
    """
    stats_from_all_models['model_name'].append(model)
    stats_from_all_models['model_params'].append(model_params)
    stats_from_all_models[f'mean_{eval_type}_accuracy'].append(mean_acc)
    stats_from_all_models[f'mean_{eval_type}_loss'].append(mean_loss)
    stats_from_all_models['std_error_accuracy'].append(std_error_acc)
    stats_from_all_models['std_error_auc'].append(std_error_loss)

def save_and_print_results(stats_from_all_models: dict, path_to_results: str, csv_name: str = "models_summary"):
    """ Save the results of all models to a csv file and print them to the console.

    Args:
        stats_from_all_models (dict): Dictionary containing the statistics of all models
        path_to_results (str): Path to the directory where the results should be saved
        csv_name (str, optional): Prefix of the csv and tex file. Defaults to "models_summary".
    """
    df_combined = pd.DataFrame(stats_from_all_models)
    df_combined.to_csv(os.path.join(path_to_results, csv_name + ".csv"), index=False)
    # drop the model_name column for the latex table
    del df_combined['model_name']
    df_combined.to_latex(os.path.join(path_to_results, csv_name + ".tex"), index=False, float_format="%.4f")
    print(f"Summary of the models saved to {os.path.join(path_to_results, csv_name)}")
    print(df_combined)

def main(patch_size: str, dataset: str, data_specifier: str, resnet_type: str, 
         annotations_dir: str, eval_type: str):
    """ Main function to summarize the results from evaluation during training of the cross-validation in a nice table.

    Args:
        patch_size (str): Size of the patches
        dataset (str): Name of the dataset
        data_specifier (str): String specifying the patches to be used, e.g top_5_patches
        resnet_type (str): Type of ResNet to use
        annotations_dir (str): Path to the directory where the different annotations files are located which is needed for the dataloader
        eval_type (str): Type of evaluation used during training. Either "patch" or "slide"
    """
    path_to_results = os.path.join("/mnt/resnet_ddp/experiments", patch_size, dataset, f"models_on_{data_specifier}", annotations_dir)
    models_to_evaluate = get_models_to_evaluate(path_to_results, resnet_type) 

    if eval_type != 'slide' and eval_type != 'patch':
        raise ValueError(f"Unknown evaluation type: {eval_type}")
    else:
        stats_from_all_models = {
            'model_name': [],
            'model_params': [], 
            f'mean_{eval_type}_accuracy': [], 
            f'mean_{eval_type}_loss': [], 
            'std_error_accuracy': [], 
            'std_error_auc': []
            }

    # extract number of folds from annotations_dir (inner_5_fold_cv_patchesPercent_100.0_shufflePatches...)
    k = int(annotations_dir.split("_")[1])
    stats_file_name = f"{k}_fold_cv_{eval_type}_stats.csv"

    for model in models_to_evaluate:
        load_model_stats_and_append_to_dict(stats_from_all_models, model, path_to_results, stats_file_name, eval_type)

    if eval_type == "slide":
        # rename key mean_slide_loss to mean_slide_auc
        stats_from_all_models['mean_slide_auc'] = stats_from_all_models.pop('mean_slide_loss')
        stats_from_all_models['std_error_accuracy'] = stats_from_all_models.pop('std_error_accuracy') # just to keep the order
        stats_from_all_models['std_error_auc'] = stats_from_all_models.pop('std_error_auc') # just to keep the order


    # Switch the dictionary to a pandas dataframe and save and print it
    save_and_print_results(stats_from_all_models, path_to_results)


if __name__ == '__main__':
    args = parse_arguments()
    main(**args)

