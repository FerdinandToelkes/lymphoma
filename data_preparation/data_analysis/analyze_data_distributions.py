import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from lymphoma.diagnosis_maps import LABELS_MAP_INT_TO_STRING




def parse_arguments() -> argparse.Namespace:
    """ Parse command line arguments. 
    
    Returns:
        args (argparse.Namespace): Command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_annotations', type=str, required=True, help='Path to annotations directory containing csv files to split into train, validation and test set.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory of a training run.')
    parser.add_argument('--path_to_infos', type=str, default='/mnt/lymphoma/infos_about_sqlite_data/metadata.csv', help='Path to metadata.csv file created by Patchcraft create_info_file. Defaults to /mnt/lymphoma/infos_about_sqlite_data/metadata.csv. Defaults to /mnt/lymphoma/infos_about_sqlite_data/metadata.csv.')
    parser.add_argument('--plots_dir', type=str, default="plots", help='Name of directory where plots will be saved within the output_dir. Defaults to "plots".')
    return parser.parse_args()


#########################################################################################################################
########################################### ANALYZE TRAIN, VAL, TEST SPLITS #############################################
#########################################################################################################################

def analyze_annotation_files_used_for_training(output_dir: str, path_to_annotations: str, plots_dir: str = "plots") -> float:
    """ Analyzes the splits .csv files used for training, validation, and testing, and plots number of slides per class and split as one bar plot. 
    
    Args:
        output_dir (str): Path to output directory of a training run.
        path_to_annotations (str): Path to annotations directory containing csv files to split into train, validation and test set.
        plots_dir (str, optional): Name of directory where plots will be saved within the output_dir. Defaults to "plots".

    Returns:
        baseline_accuracy (float): Baseline accuracy for test data resulting from splits.
    """
    # analyze the splits.csv file
    df_train = pd.read_csv(os.path.join(path_to_annotations, 'train.csv'))
    df_val = pd.read_csv(os.path.join(path_to_annotations, 'val.csv'))
    df_test = pd.read_csv(os.path.join(path_to_annotations, 'test.csv'))
    print(f"Length of train.csv (one stain): {len(df_train)}")
    print(f"Length of val.csv (one stain):   {len(df_val)}")
    print(f"Length of test.csv (one stain):  {len(df_test)} \n")

    # get number of patches per class and split
    nr_of_patches_per_class_per_split = get_nr_of_patches_per_class_per_split(df_train, df_val, df_test)
    print(f"Number of patches per class per split: {nr_of_patches_per_class_per_split}\n")

    # compute a baseline accuracy for the test set by predicting the majority class
    classes = list(nr_of_patches_per_class_per_split.keys())
    majority_index = np.argmax(list(nr_of_patches_per_class_per_split['%s' % c]['train'] for c in classes))
    majority_class = classes[majority_index]
    baseline_accuracy = nr_of_patches_per_class_per_split['%s' % majority_class]['test'] / np.sum(list(nr_of_patches_per_class_per_split['%s' % c]['test'] for c in classes))

    # plot number of patches per class and split as one bar plot
    if not os.path.exists(os.path.join(output_dir, plots_dir)):
        print(f"Creating directory {os.path.join(output_dir, plots_dir)}")
        os.makedirs(os.path.join(output_dir, plots_dir))
    name = os.path.join(output_dir, plots_dir, 'patches_per_class_per_split.png')
    plot_patches_per_class_per_split(nr_of_patches_per_class_per_split, baseline_accuracy, dpi=500, name=name)

    return baseline_accuracy


#########################################################################################################################
################################################## Helper functions #####################################################
#########################################################################################################################

def get_nr_of_patches_per_class_per_split(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """ Count number of slides per class and split and return as dict.
    
    Args:
        df_train (pd.DataFrame): Dataframe containing the training split.
        df_val (pd.DataFrame): Dataframe containing the validation split.
        df_test (pd.DataFrame): Dataframe containing the test split.

    Returns:
        nr_of_patches_per_class_per_split (dict): Dictionary containing the number of patches per class per split.
    """
    # get all different classes and set up dict, note that in test set, at least 5 slides per class 
    # are present if possible -> not every set contains all classes!
    classes = pd.concat([df_train['label'], df_val['label'], df_test['label']]).unique()
    classes.sort()
    nr_of_patches_per_class_per_split = {'%s' % c: None for c in classes}
    for c in classes:
        nr_of_patches_per_class_per_split['%s' % c] = {'train': None, 'validation': None, 'test': None}
        nr_of_patches_per_class_per_split['%s' % c]['train'] = len(df_train[df_train['label'] == c])
        nr_of_patches_per_class_per_split['%s' % c]['validation'] = len(df_val[df_val['label'] == c])
        nr_of_patches_per_class_per_split['%s' % c]['test'] = len(df_test[df_test['label'] == c])
    # sanity check
    if np.sum(list(nr_of_patches_per_class_per_split['%s' % c]['train'] for c in classes)) + \
       np.sum(list(nr_of_patches_per_class_per_split['%s' % c]['validation'] for c in classes)) + \
       np.sum(list(nr_of_patches_per_class_per_split['%s' % c]['test'] for c in classes)) != len(df_train) + len(df_val) + len(df_test):
        raise ValueError(f"The number of slides counted per class per split does not match the total number of slides.")
    return nr_of_patches_per_class_per_split

def plot_patches_per_class_per_split(nr_of_patches_per_class_per_split: dict, baseline_accuracy: float, 
                                     dpi: int = 500, name: str = 'plots/nr_of_patches_per_class_per_split.png'):
    """ Plots the number of patches per class per split as one bar plot. 
    
    Args:
        nr_of_patches_per_class_per_split (dict): Dictionary containing the number of patches per class per split.
        baseline_accuracy (float): Baseline accuracy for test data resulting from splits.
        dpi (int, optional): Dots per inch for the plot. Defaults to 500.
        name (str, optional): Name of the plot file. Defaults to 'plots/nr_of_patches_per_class_per_split.png'.
    """
    # get all different classes
    classes = list(nr_of_patches_per_class_per_split.keys())
    classes.sort()
    # plot three bars per class in one bar plot
    width = 0.25
    x_pos = np.arange(len(classes))
    x_labels = [LABELS_MAP_INT_TO_STRING[int(c)] for c in classes]
    _, ax = plt.subplots()
    ax.bar(x_pos - width, list(nr_of_patches_per_class_per_split['%s' % c]['train'] for c in classes), width, label='train')
    ax.bar(x_pos, list(nr_of_patches_per_class_per_split['%s' % c]['validation'] for c in classes), width, label='validation')
    ax.bar(x_pos + width, list(nr_of_patches_per_class_per_split['%s' % c]['test'] for c in classes), width, label='test')
    ax.set_title('Number of patches per class per split (for HE stain)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend()

    # add baseline accuracy below plot
    ax.text(0.5, -0.12, f"=> Baseline accuracy for test set: {baseline_accuracy*100:.2f} % instead of 1/{str(len(classes))} = {100/len(classes):.2f} %", transform=ax.transAxes, ha="center")

    # save plot
    print(f"Saving plot to {name}")
    plt.savefig(name, dpi=dpi)
    plt.close()

#########################################################################################################################
########################################### ANALYZE ORIGINAL SQLITE DATA ################################################
#########################################################################################################################


def analyze_original_sqlite_data(path_to_infos: str, plots_dir: str = "plots"):
    """ Analyzes the metadata.csv file created by blobyfire create_info_file and plots number of slides per stain 
    and number of slides per diagnosis per stain as bar plots. 
    
    Args:
        path_to_infos (str): Path to metadata.csv file.
        plots_dir (str, optional): Name of directory where plots will be saved within the output_dir. Defaults to "plots".
    """
    df_infos = pd.read_csv(path_to_infos)
    nr_of_slides = len(df_infos)
    print(f"Total number of slides from all stains : {nr_of_slides}")

    # get number of slides per stain and number of slides per diagnosis per stain
    nr_of_slides_per_stain = get_nr_of_slides_per_stain(df_infos, nr_of_slides)
    nr_of_slides_per_diagnosis_per_stain = get_nr_of_slides_per_diagnosis_per_stain(df_infos, nr_of_slides)
    print("Number of slides per stain:", nr_of_slides_per_stain)
    print("Number of slides per diagnosis per stain:", nr_of_slides_per_diagnosis_per_stain)

    # plot number of slides per stain and number of slides per diagnosis per stain as bar plots
    # remove the last part after the last / of the path of path_to_infos to get the base name
    base_name = os.path.dirname(path_to_infos)
    name = os.path.join(base_name, plots_dir, 'nr_of_slides_per_stain.png')
    plot_slides_per_stain(nr_of_slides_per_stain, dpi=500, name=name)
    name = os.path.join(base_name, plots_dir, 'slides_per_diagnosis_per_stain.png')
    plot_slides_per_diagnosis_per_stain(df_infos, nr_of_slides_per_diagnosis_per_stain, dpi=500, name=name)


#########################################################################################################################
################################################## Helper functions #####################################################
#########################################################################################################################


def get_nr_of_slides_per_stain(df_infos: pd.DataFrame, total_nr_of_slides: int) -> dict:
    """ Get number of slides per stain and return as dict. 
    
    Args:
        df_infos (pd.DataFrame): Dataframe containing the metadata.
        total_nr_of_slides (int): Total number of slides.

    Returns:
        nr_of_slides_per_stain (dict): Dictionary containing the number of slides per stain.
    """
    # get all stains and set up dict
    stains = df_infos['stain'].unique()
    nr_of_slides_per_stain = {'%s' % s: None for s in stains}
    count = 0
    for s in stains:
        if s != s: # check for nan
            nr_of_slides = df_infos[df_infos['stain'].isna()].index.size
        else:
            nr_of_slides = df_infos[df_infos['stain'] == s].index.size
        count += nr_of_slides
        nr_of_slides_per_stain['%s' % s] = nr_of_slides
    # sanity check
    if count != total_nr_of_slides:
        raise ValueError(f"There were {count} slides counted, but there should be {total_nr_of_slides} slides.")
    return nr_of_slides_per_stain


def get_nr_of_slides_per_diagnosis_per_stain(df_infos: pd.DataFrame, total_nr_of_slides: int) -> dict:
    """ Get number of slides per diagnosis per stain and return as dict. 
    
    Args:
        df_infos (pd.DataFrame): Dataframe containing the metadata.
        total_nr_of_slides (int): Total number of slides.

    Returns:
        nr_of_slides_per_diagnosis_per_stain (dict): Dictionary containing the number of slides per diagnosis per stain.
    """
    # get all different stains and set up dict
    stains = df_infos['stain'].unique()
    nr_of_slides_per_diagnosis_per_stain = {'%s' % s: None for s in stains}
    count = 0
    for s in stains:
        # check for nan
        if s != s:
            df_stain = df_infos[df_infos['stain'].isna()]
        else:
            df_stain = df_infos[df_infos['stain'] == s]
        # get all different diagnoses for the current stain and set up dict
        diagnoses = df_stain['diagnosis'].unique()
        diagnoses.sort()
        nr_of_slides_per_diagnosis_per_stain['%s' % s] = {'%s' % d: None for d in diagnoses}
        for d in diagnoses:
            # check for nan
            if d != d:
                nr_of_slides = df_stain[df_stain['diagnosis'].isna()].index.size
            else:
                nr_of_slides = df_stain[df_stain['diagnosis'] == d].index.size
            count += nr_of_slides
            nr_of_slides_per_diagnosis_per_stain['%s' % s]['%s' % d] = nr_of_slides
    # sanity check
    if count != total_nr_of_slides:
        raise ValueError(f"There were {count} slides counted, but there should be {total_nr_of_slides} slides.")
    return nr_of_slides_per_diagnosis_per_stain


def plot_slides_per_stain(nr_of_slides_per_stain: dict, dpi: int = 500, name: str = 'nr_of_slides_per_stain.png'):
    """ Plots the number of slides per stain as a bar plot. 
    
    Args:
        nr_of_slides_per_stain (dict): Dictionary containing the number of slides per stain.
        dpi (int, optional): Dots per inch for the plot. Defaults to 500.
        name (str, optional): Name of the plot file. Defaults to 'nr_of_slides_per_stain.png
    """
    plt.bar(nr_of_slides_per_stain.keys(), nr_of_slides_per_stain.values())
    plt.title('Number of slides per stain')
    plt.xlabel('Stain')
    plt.ylabel('Number of slides')
    print(f"Saving plot to {name}")
    if not os.path.exists(os.path.dirname(name)):
        print(f"Creating directory {os.path.dirname(name)}")
        os.makedirs(os.path.dirname(name))
    if os.path.exists(name):
        print(f"Overwriting existing file {name}.")
    plt.savefig(name, dpi=dpi)
    plt.close()

def plot_slides_per_diagnosis_per_stain(df_infos: pd.DataFrame, nr_of_slides_per_diagnosis_per_stain: dict, 
                                        dpi: int = 500, name: str = 'nr_of_slides_per_diagnosis_per_stain.png'):
    """ Plots the number of slides per diagnosis per stain as grid plot of bar plots. NaN stain is not plotted. 
    
    Args:
        df_infos (pd.DataFrame): Dataframe containing the metadata.
        nr_of_slides_per_diagnosis_per_stain (dict): Dictionary containing the number of slides per diagnosis per stain.
        dpi (int, optional): Dots per inch for the plot. Defaults to 500.
        name (str, optional): Name of the plot file. Defaults to 'nr_of_slides_per_diagnosis_per_stain.png'.
    """
    # get all different stains that are not nan 
    stains = df_infos['stain'].unique()
    stains_without_nan = [s for s in stains if s == s]

    if stains_without_nan == []:
        raise ValueError("No stains without nan found in the metadata file.")
    
    # set up single plot or grid plot
    if len(stains_without_nan) == 1:
        # set up single plot
        plt.bar(nr_of_slides_per_diagnosis_per_stain['%s' % stains_without_nan[0]].keys(), nr_of_slides_per_diagnosis_per_stain['%s' % stains_without_nan[0]].values())
        plt.title(f'Number of slides per diagnosis per stain (nan stain is not plotted)')
        plt.xlabel('Diagnosis')
        plt.ylabel('Number of slides')
    else:
        # set up grid plot
        rows = 2
        cols = int(np.ceil(len(stains_without_nan) / rows))
        fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
        fig.suptitle('Number of slides per diagnosis per stain (nan stain is not plotted)')

        for i, s in enumerate(stains_without_nan):
            axs[i//cols, i%cols].bar(nr_of_slides_per_diagnosis_per_stain['%s' % s].keys(), nr_of_slides_per_diagnosis_per_stain['%s' % s].values())
            axs[i//cols, i%cols].set_title(f'Stain: {s}')
            axs[i//cols, i%cols].set_xlabel('Diagnosis')
            axs[i//cols, i%cols].set_ylabel('Number of slides')
    print(f"Saving plot to {name}")
    if not os.path.exists(os.path.dirname(name)):
        print(f"Creating directory {os.path.dirname(name)}")
        os.makedirs(os.path.dirname(name))
    if os.path.exists(name):
        print(f"Overwriting existing file {name}.")
    plt.savefig(name, dpi=dpi)
    plt.close()


#########################################################################################################################

def main(path_to_annotations: str, output_dir: str, path_to_infos: str):
    """ Main function to analyze the data distributions.

    Args:
        path_to_annotations (str): Path to annotations directory containing csv files to split into train, validation and test set.
        output_dir (str): Path to output directory of a training run.
        path_to_infos (str): Path to metadata.csv file created by Patchcraft create_info_file.
    """
    # update path to the output directory
    annotations_dir = path_to_annotations.split('/')[-1] # get the last part of the path
    output_dir = os.path.join(output_dir, annotations_dir)

    # check if the annotations directory exists
    if not os.path.exists(path_to_annotations):
        raise Exception(f"Annotations directory does not exist at this path: {path_to_annotations}")

    # analyze the splits.csv file used for training and plot number of slides per class and split as one bar plot
    baseline_accuracy = analyze_annotation_files_used_for_training(output_dir, path_to_annotations)
    print(f"Baseline accuracy for test data resulting from splits: {baseline_accuracy*100:.2f} % \n")
    
    # check if the metadata.csv file exists in the specified path
    if not os.path.exists(path_to_infos):
        raise Exception(f"Metadata file does not exist in the specified path: {path_to_infos}. Use the 'create_info_file' command to create it.")
    
    # analyze the data directory, i.e., count number of slides per class and stain and plot as bar plots
    analyze_original_sqlite_data(path_to_infos, output_dir)
    
       

if __name__ == "__main__":
    args = parse_arguments()
    main(args.path_to_annotations, args.output_dir, args.path_to_infos)
    