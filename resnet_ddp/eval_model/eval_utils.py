import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
import os

from sklearn.metrics import auc, roc_curve


def compute_class_accuracies(correct_pred: dict, total_occurances: dict, classes: list, 
                             eval_type: str = "Patch") -> dict:
    """ Computes the accuracy for each class by averaging over all gpus.
    
    Args:
        correct_pred (dict): Dictionary containing the correct predictions for each class.
        total_occurances (dict): Dictionary containing the total predictions for each class.
        classes (list): List of class names.
        eval_type (str, optional): Type of evaluation, either "Patch" or "Slide". Defaults to "Patch".
    """
    # setup dictionary for class accuracies and compute the accuracy for each class -> we want to save the correct predictions, total predictions and the accuracy for each class
    class_accuracies = {classname: None for classname in classes} 
    for classname, correct_count in correct_pred.items():
        # dist.all_reduce(correct_count, op=dist.ReduceOp.SUM) 
        # only for patch necessary since the two variables are already summed up in slide evaluation
        if eval_type == "Patch":
            # sum the correct predictions and total predictions over all gpus
            dist.all_reduce(correct_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_occurances[classname], op=dist.ReduceOp.SUM)
        # compute the accuracy for current class
        if total_occurances[classname] == 0: # avoid division by zero
            accuracy = 0
        else:
            accuracy = 100 * float(correct_count) / total_occurances[classname]
        class_accuracies[classname] = {"corrects": correct_count, "total": total_occurances[classname], "accuracy": accuracy}
    return class_accuracies


def add_total_row(df: pd.DataFrame) -> pd.DataFrame:
    """ Add row named 'Total' with total corrects and total predictions and resulting accuracy. 
    
    Args:
        df (pd.DataFrame): Dataframe containing the accuracies for each class.

    Returns:
        pd.DataFrame: Dataframe with the row 'Total' added.
    """
    totals = df[['corrects', 'total predictions']].sum()
    totals['class'] = 'Total'
    totals['accuracy'] = np.round((totals['corrects'] / totals['total predictions'] * 100), 2)
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    return df

def save_accuracies(df: pd.DataFrame, output_dir: str, eval_type: str, gpu_id: int, k: int = -1):
    """ Saves the dataframe with the accuracies to a CSV file in a seperate output directory. 
    
    Args:
        df (pd.DataFrame): Dataframe containing the accuracies for each class.
        output_dir (str): Path to the output directory where the CSV file should be saved.
        eval_type (str): Type of evaluation, either "Patch" or "Slide".
        gpu_id (int): ID of the GPU.
        k (int, optional): Fold number. Defaults to -1.
    """
    # save the dataframe to a csv file
    if k != -1:
        save_path = os.path.join(output_dir, f'{eval_type.lower()}_accuracies_fold_{k}.csv')
    else:
        save_path = os.path.join(output_dir, f'{eval_type.lower()}_accuracies.csv')
    print(f"[GPU{gpu_id}] Saving accuracies to {save_path}")
    df.to_csv(save_path, index=False)


def summarize_accuracies_to_df(class_accuracies: dict) -> tuple:
    """ Converts the accuracies to floats, adds total accuracy and returns them as pandas dataframe. 
    
    Args:
        class_accuracies (dict): Dictionary containing the accuracies for each class.

    Returns:
        tuple: Tuple containing the dataframe with the accuracies and the total accuracy.
    """
    # convert the accuracies to floats if they are not already floats and split the accuracies into three lists
    df_class_accuracies = []
    for label, stats in class_accuracies.items():
        # convert tensors to floats if necessary
        for stat, value in stats.items():
            if isinstance(value, torch.Tensor):
                class_accuracies[label][stat] = round(value.cpu().item(), 2)
        # change form of dictionary
        df_class_accuracies.append([label, stats['corrects'], stats['total'], stats['accuracy']])

    # convert the dictionary to a pandas dataframe with four columns and save it to a csv file
    df = pd.DataFrame(df_class_accuracies, columns=['class', 'corrects', 'total predictions', 'accuracy'])
    # add row named 'Total' with total corrects and total predictions and resulting accuracy
    df = add_total_row(df)
    # return total accuracy: column 'accuracy' of the row named 'Total'
    total_acc = df[df['class'] == 'Total']['accuracy'].values[0]
    return df, total_acc
    


def compute_auc(probs_and_labels: dict, labels_map_str_to_int: dict) -> tuple:
    """ Computes the AUC for each class and averages it over all classes. 
    
    Args:
        probs_and_labels (dict): Dictionary containing the probabilities and labels.
        labels_map_str_to_int (dict): Dictionary mapping class names to integers.

    Returns:
        tuple: Tuple containing the average AUC and the AUC values for each class.
    """
    # convert the probability and label lists to numpy arrays
    probabilities = np.array(probs_and_labels['probs']) # dim = (num_samples, num_classes)
    labels = np.array(probs_and_labels['labels']) # dim = (num_samples,)

    # setup dictionary for the AUC values and compute the AUC for each class
    classes = list(labels_map_str_to_int.values())
    classes_ints = list(labels_map_str_to_int.keys())
    auc_values = {}
    for classname, class_int in zip(classes, classes_ints):
        # get the probabilities and labels for the current class
        # auc is a metric for binary classification, so we need to compute the auc for each class separately
        # and transform the multi-class classification problem into multiple binary classification problems
        class_probs = probabilities[:, classes.index(classname)] # get the probabilities for the current class
        class_labels = [1 if label == class_int else 0 for label in labels] # convert the labels to binary labels
        if sum(class_labels) == 0: # skip classes with no samples
            print(f"Skipping class {classname} since it does not appear in the ground truth labels.")
            continue
        # compute the AUC for the current class -> compute it like that to be consistent with script for plotting it
        fpr, tpr, thresholds = roc_curve(class_labels, class_probs)
        auc_score = auc(fpr, tpr)
        auc_values[classname] = auc_score
    # average the AUC values over all classes that were not skipped
    print(f"AUC values: {auc_values}")
    auc_array = np.array(list(auc_values.values()))
    avg_auc = np.mean(auc_array) 
    return avg_auc, auc_values
    


def save_confusion_matrix(np_confusion_matrix: np.ndarray, classes: list, output_dir: str, 
                          eval_type: str, gpu_id: int, k: int = -1):
    """ Saves the confusion matrix to a CSV file in a seperate output directory. 
    
    Args:
        np_confusion_matrix (np.ndarray): Confusion matrix as numpy array.
        classes (list): List of class names.
        output_dir (str): Path to the output directory where the CSV file should be saved.
        eval_type (str): Type of evaluation, either "Patch" or "Slide".
        gpu_id (int): ID of the GPU.
        k (int, optional): Fold number. Defaults to -1.
    """
    # convert the confusion matrix to a pandas dataframe
    df = pd.DataFrame(np_confusion_matrix, columns=classes) #, index=self.classes
    # add an extra column which contains the class names and drop the index
    df.insert(0, 'act. class (rows)/pred. class(cols)', classes)
    df.set_index('act. class (rows)/pred. class(cols)', inplace=True)

    # save the dataframe to a csv file
    if k != -1:
        plot_path = os.path.join(output_dir, f'{eval_type.lower()}_confusion_matrix_fold_{k}.csv')
    else:
        plot_path = os.path.join(output_dir, f'{eval_type.lower()}_confusion_matrix.csv')
    print(f"[GPU{gpu_id}] Saving confusion matrix to {plot_path}")
    df.to_csv(plot_path, index=False)

