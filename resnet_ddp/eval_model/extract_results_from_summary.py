import pandas as pd
import argparse
import os

from lymphoma.resnet_ddp.eval_model.summarize_cv_results import save_and_print_results

# script to extract desired results from the cross validations summary file such as only results for weighted loss = True

# run script locally from code directory with:
# python3 -m lymphoma.resnet_ddp.eval_model.extract_results_from_summary --patch_size 1024um -ds kiel --data_specifier patches --data_dir data_dir --weighted_loss 

def parse_arguments() -> dict:
    """ Parse the arguments given to the script. 
    
    Returns:
        dict: A dictionary containing the arguments as key-value pairs.
    """
    parser = argparse.ArgumentParser(description='Extract desired results from the cross validations summary file.')
    # arguments needed to specify the path to the summary file
    parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
    parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply", "munich" or "all_data". (default: kiel)')
    parser.add_argument('--data_specifier', default='patches', type=str, help='String specifying the patches to be used, e.g top_5_patches (default: "patches")')
    parser.add_argument('--annotations_dir', default="inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0", type=str, help='Path to the directory where the different annotations files are located which is needed for the dataloader. (default: inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0)')
    # arguments needed to specify the desired results
    parser.add_argument("--data_dir", default="embeddings_dir", type=str, help="Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'. (default: embeddings_dir)")
    parser.add_argument('--weighted_loss', default=False, action='store_true', help='Whether to use weighted loss or not. (default: False)')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='Number of warmup epochs. (default: 0)')
    args = parser.parse_args()
    return dict(vars(args))

def main(patch_size: str, dataset: str, data_specifier: str, annotations_dir: str, 
         data_dir: str, weighted_loss: bool, warmup_epochs: int):
    """ Extract desired results from the cross validations summary file.

    Args:
        patch_size (str): Size of the patches.
        dataset (str): Name of the dataset.
        data_specifier (str): String specifying the patches to be used.
        annotations_dir (str): Path to the directory where the different annotations files are located which is needed for the dataloader.
        data_dir (str): Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'.
        weighted_loss (bool): Whether to use weighted loss or not.
        warmup_epochs (int): Number of warmup epochs.
    """
    path = f"/Users/ferdinandtolkes/code/lymphoma/resnet_ddp/experiments/{patch_size}/{dataset}/models_on_{data_specifier}/{annotations_dir}"
    df = pd.read_csv(os.path.join(path, "models_summary.csv"))
    print(df)
    # drop rows where ",dd={string}," in the model_name column, does not match the data_dir 
    data_dir_pattern = f",dd={data_dir},"
    df = df[df["model_name"].str.contains(data_dir_pattern)]
    print(f"Results for data_dir: {data_dir}")
    print(df)
    wl_pattern = f",wl={weighted_loss},"
    df = df[df["model_name"].str.contains(wl_pattern)]
    # for many wu=0 models, the warmup_epochs are not included in the model_name, so we need to filter them out
    if warmup_epochs != 0 and "wu="  in df["model_name"].iloc[0]:
        wu_pattern = f",wu={warmup_epochs},"
        df = df[df["model_name"].str.contains(wu_pattern)]
    print(f"Filtered results for weighted_loss: {weighted_loss} and warmup_epochs: {warmup_epochs}")
    print(df)
    save_and_print_results(df, path, f"results_{data_dir}_wl_{weighted_loss}_wu_{warmup_epochs}")

if __name__ == "__main__":
    args = parse_arguments()
    main(**args)