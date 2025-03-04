import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, log_loss

from .eval_utils import compute_auc
from lymphoma.resnet_ddp.utils import get_best_model_dir
from lymphoma.diagnosis_maps import LABELS_MAP_INT_TO_STRING

# script for visualizing the results of the model evaluation on the test data for local usage because its easier to modify the plots
# execute from code directory with:
"""
python3 -m lymphoma.resnet_ddp.eval_model.plot_saved_results --patch_size="1024um" --dataset="marr" --data_dir="embeddings_dir" --data_specifier=patches --eval_dataset="kiel" --resnet_dir="RESNET18,dd=embeddings_dir,e=50,lr=0.001,bs=256,wl=True,ls=0.1,p=10,s=42" --excluded_class="HL" 
"""

def set_plotting_config(fontsize: int = 10, aspect_ratio: float = 1.618, 
                        width_fraction: float = 1.0, text_usetex: bool = False):
        """ Set global plotting configuration for Matplotlib and Seaborn. 
        
        Args:   
            fontsize (int, optional): Font size for all elements. (default: 10)
            aspect_ratio (float, optional): Aspect ratio for the plot. (default: 1.618)
            width_fraction (float, optional): Width fraction of the plot. (default: 1.0)
            text_usetex (bool, optional): Whether to use LaTeX for text rendering. (default: False)
        """

        latex_text_width_in_pt = 345  # LaTeX text width in points for article class
        latex_text_width_in_in = width_fraction * latex_text_width_in_pt / 72  # Convert pt to inches
        scale_factor = width_fraction + 0.25  if width_fraction < 1.0 else 1.0


        # Set Matplotlib rcParams
        plt.rcParams.update({
            "font.family": "lmodern" if text_usetex else "sans-serif",
            "text.usetex": text_usetex,
            'font.size': fontsize * scale_factor,  
            'text.latex.preamble': r'\usepackage{lmodern}',
            "axes.labelsize": fontsize * scale_factor,
            "xtick.labelsize": (fontsize - 2) * scale_factor,
            "ytick.labelsize": (fontsize - 2) * scale_factor,
            "legend.fontsize": (fontsize - 2) * scale_factor,
            "axes.linewidth": 0.8 * scale_factor,
            "lines.linewidth": 0.8 * scale_factor,
            "grid.linewidth": 0.6 * scale_factor,
            'lines.markersize': 5 * width_fraction,
            "figure.autolayout": True,
            "figure.figsize": (latex_text_width_in_in, latex_text_width_in_in / aspect_ratio),
        }) 

        # Set color palette
        sns.set_palette("colorblind")

class ModelEvaluator:
    """ Class for evaluating the model on the test data and visualizing the results. """
    def __init__(self, patch_size: str, dataset: str, eval_dataset: str, excluded_class: str, data_dir: str, 
                 data_specifier: str, annotations_dir: str, resnet_dir: str = None):
        self.patch_size = patch_size
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.excluded_class = excluded_class
        self.data_dir = data_dir
        self.data_specifier = data_specifier
        self.annotations_dir = annotations_dir
        self.model_path = f"/Users/ferdinandtolkes/code/lymphoma/resnet_ddp/experiments/{patch_size}/{dataset}/models_on_{data_specifier}/{annotations_dir}"
        if resnet_dir is not None:
            self.resnet_dir = resnet_dir
        else:
            self.resnet_dir, _ = get_best_model_dir(self.model_path, data_dir)
        self.path = os.path.join(self.model_path, self.resnet_dir)
        self.plot_path = os.path.join(self.path, self.eval_dataset, "plots")
        os.makedirs(self.plot_path, exist_ok=True)
        self.nr_of_folds = int(annotations_dir.split("_")[1])
        self.not_available_classes = self.get_not_available_classes(eval_dataset)

    @staticmethod
    def parse_arguments() -> dict:
        """ Parse command-line arguments and return them as a dictionary.

        Returns:
            dict: A dictionary containing the parsed command-line arguments.
        """
        parser = argparse.ArgumentParser(description='Visualize final results of model evaluation on the test data')
        parser.add_argument('--patch_size', default='1024um', type=str, help='Size of the patches. (default: 1024um)')
        parser.add_argument('-ds', '--dataset', default='kiel', type=str, help='Name of the dataset, can be "kiel", "swiss_1", "swiss_2", "multiply", "marr" or "all_data". (default: kiel)')
        parser.add_argument('-eds', '--eval_dataset', default='kiel', type=str, help='Name of the dataset to be used for the test data, note that its possible to eval the models on datasets they have not been trained on (default: kiel)')
        parser.add_argument('--excluded_class', default='', type=str, help='Class to be excluded from the evaluation. (default: "")')
        parser.add_argument('--data_specifier', default='patches', type=str, help='String specifying the patches to be used, e.g top_5_patches (default: "patches")')
        parser.add_argument("--data_dir", default="embeddings_dir", type=str, help="Name of the directory containing the data: 'data_dir', 'embeddings_dir' or 'embeddings_dir_not_normalized'. (default: embeddings_dir)")
        parser.add_argument('--resnet_dir', default=None, type=str, help='Name of the directory containing the model to be evaluated. If None, the best model will be used. (default: None)')
        parser.add_argument('--annotations_dir', default="inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0", type=str, help='Path to the directory where the different annotations files are located which is needed for the dataloader. (default: inner_5_fold_cv_patchesPercent_100.0_shufflePatches_True_shuffleDataframes_False_tumorThreshold_0_seed_0)')
        args = parser.parse_args()
        return dict(vars(args))
    
    
    def get_not_available_classes(self, eval_dataset: str) -> list:
        """ Get the classes that are not available in the eval data as integers.

        Args:
            eval_dataset (str): Name of the dataset to be used for the test data.

        Returns:
            list: List of classes that are not available in the eval data as integers.
        """
        if eval_dataset == "kiel":
            not_available_classes = []
        elif eval_dataset == "swiss_1" or eval_dataset == "swiss_2" or eval_dataset == "multiply":
            not_available_classes = ["HL", "FL", "DLBCL", "LTDS", "CLL"]
        elif eval_dataset == "munich":
            not_available_classes = ["HL"] 
        NEW_LABELS_MAP_STRING_TO_INT = {label: i for i, label in enumerate(not_available_classes)}
        not_available_classes = [NEW_LABELS_MAP_STRING_TO_INT[label] for label in not_available_classes]
        return not_available_classes

    def load_eval_results_and_visualize(self, base_name: str, level: str, do_cm_plot: bool = True, 
                                        do_roc_plot: bool = True):
        """ Load the model outputs from test data and corresponding ground truths, get the predictions and visualize them. 
        
        Args:
            base_name (str): Base name of the file containing the model outputs and ground truths.
            level (str): The level of the predictions (e.g., "slide" or "patch").
            do_cm_plot (bool): Whether to plot the confusion matrix. (default: True)
            do_roc_plot (bool): Whether to plot the ROC curves. (default: True)
        """
        # Load the probabilities, labels, and compute the predicted labels
        probs_and_labels, _, ground_truths, predicted_labels = self.process_predictions(base_name, level)

        # set number of classes that actually exist in the data
        classes_in_ground_truth, all_classes = self.check_for_unique_classes_and_labels(ground_truths, predicted_labels)

        # Compute and plot the confusion matrix
        if do_cm_plot:
            # cm = confusion_matrix(ground_truths, predicted_labels, labels=classes_in_ground_truth)
            cm = confusion_matrix(ground_truths, predicted_labels, labels=all_classes)
            # str_labels = [LABELS_MAP_INT_TO_STRING[label] for label in classes_in_ground_truth]
            str_labels = [LABELS_MAP_INT_TO_STRING[label] for label in all_classes]
            set_plotting_config(fontsize=8, aspect_ratio=1/1, width_fraction=0.5, text_usetex=True)
            self.plot_confusion_matrix(cm, str_labels, False, level)

        # Compute and plot the one-vs-all ROC curves
        if do_roc_plot:
            roc_curves = self.compute_roc_curves(probs_and_labels, LABELS_MAP_INT_TO_STRING)
            set_plotting_config(fontsize=8, aspect_ratio=1/1, width_fraction=0.5, text_usetex=True)
            self.plot_roc_curves(roc_curves, level)


    def process_predictions(self, base_name: str, level: int, prediction_function: callable = np.argmax) -> tuple: 
        """ Load probabilities and labels, compute predicted labels using a custom prediction function,
            and return all relevant data.

        Args:
            base_name (str): The base name for the data.
            level (int): The level of detail for the data.
            prediction_function (callable, optional): A function to compute predictions from probabilities.
                Defaults to numpy's argmax.

        Returns:
            tuple: A tuple containing:
                - probs_and_labels (dict): The raw probabilities and labels dictionary.
                - model_outputs (np.ndarray): The model's raw probability outputs.
                - ground_truths (np.ndarray): The ground truth labels.
                - predicted_labels (np.ndarray): The predicted class labels.
        """
        # Load the probabilities and labels
        probs_and_labels = self.get_probs_and_labels(base_name, level=level)
        model_outputs = probs_and_labels['probs']
        ground_truths = probs_and_labels['labels']
        # Compute predictions using the provided prediction function
        predicted_labels = prediction_function(model_outputs, axis=1)
        return probs_and_labels, model_outputs, ground_truths, predicted_labels


    def check_for_unique_classes_and_labels(self, ground_truths: np.ndarray, 
                                            predicted_labels: np.ndarray) -> np.ndarray:
        """ Check if the number of classes in ground truth and predictions match and 
            return the classes that actually exist in the data. 
        
        Args:
            ground_truths (np.ndarray): Ground truth labels.
            predicted_labels (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: The classes that actually exist in the data.
        """
        classes_in_ground_truth = np.unique(ground_truths)
        classes_in_predictions = np.unique(predicted_labels)
        all_classes = np.unique(np.concatenate((classes_in_ground_truth, classes_in_predictions)))

        return classes_in_ground_truth, all_classes

    def get_probs_and_labels(self, base_name: str, level: str) -> np.ndarray:
        """ Load the probabilities and labels from the path.

        Args:
            base_name (str): Base name of the file containing the probabilities and labels.
            level (str): The level of the predictions (e.g., "slide" or "patch").

        Returns:
            np.ndarray: The probabilities and labels.
        """
        if self.excluded_class != "":
            level = f"{level}_without_{self.excluded_class}"
        name = base_name.split("/")[0] + "/" + f'{level}_{base_name.split("/")[1]}'
        path_to_probs_and_labels = os.path.join(self.path, name)
        if not os.path.exists(path_to_probs_and_labels):
            raise ValueError(f"Path to probs and labels does not exist: {path_to_probs_and_labels}. Try running test_best_model_ensemble.py first.")
        
        probs_and_labels = np.load(path_to_probs_and_labels)
        return probs_and_labels

    def plot_confusion_matrix(self, cm: np.ndarray, str_labels: list, normalize: bool, level: str, cmap=None):
        """ Plot a confusion matrix with Seaborn and Matplotlib.

        Args:
            cm (np.ndarray): A confusion matrix.
            str_labels (list): A list of class labels.
            normalize (bool): Whether to normalize the confusion matrix.
            level (str): The level of the predictions (e.g., "slide" or "patch").
            cmap (str, optional): The colormap to use. Defaults to None.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create a figure 
        plt.figure()
        
        # Create the heatmap with refined settings
        ax = sns.heatmap(
            cm,
            cmap="Blues" if cmap is None else cmap, # Use the Blues colormap if none is provided
            annot=True,                        # Annotate cells with values else "d",  
            fmt = '.2f' if normalize else 'd', # Format annotation with 2 decimal places                      
            cbar=False,                        # Disable the color bar
            xticklabels=str_labels,            # Class labels for x-axis
            yticklabels=str_labels,            # Class labels for y-axis
            square=True,                       # Make heatmap cells square
            annot_kws={"fontsize": plt.rcParams["font.size"]},  # Adjust annotation font size
        )

        # Set font size for axis labels (tick labels)
        plt.xticks(fontsize=plt.rcParams["font.size"] - 2)  
        plt.yticks(fontsize=plt.rcParams["font.size"] - 2)  

        # Add axes labels
        plt.xlabel("Predicted Labels") 
        plt.ylabel("True Labels")

        # Move x-axis labels to the top
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        # Add minor tick lines for precision alignment
        ax.tick_params(axis='both', which='both', length=0)  # Hide tick marks
        for _, spine in ax.spines.items():  # Hide spines for a clean look
            spine.set_visible(False)

        plt.tight_layout()
        save_path = os.path.join(self.plot_path, f"{level}_confusion_matrix.pdf")
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Confusion matrix saved to: {save_path}")



    def compute_roc_curves(self, probs_and_labels: dict, labels_map_str_to_int: dict) -> tuple:
        """ Computes the AUC for each class and averages it over all classes. 
        
        Args:
            probs_and_labels (dict): A dictionary containing the probabilities and labels.
            labels_map_str_to_int (dict): A dictionary mapping class names to integers.

        Returns:
            dict: A dictionary containing the ROC curves for each
                class with their corresponding AUC values.
        """
        # convert the probability and label lists to numpy arrays
        probabilities = np.array(probs_and_labels['probs']) # dim = (num_samples, num_classes)
        labels = np.array(probs_and_labels['labels']) # dim = (num_samples,)

        # setup dictionary for the AUC values and compute the AUC for each class
        classes = list(labels_map_str_to_int.values())
        classes_ints = list(labels_map_str_to_int.keys())
        roc_curves = {}
        for classname, class_int in zip(classes, classes_ints):
            # get the probabilities and labels for the current class
            # auc is a metric for binary classification, so we need to compute the auc for each class separately
            # and transform the multi-class classification problem into multiple binary classification problems
            class_probs = probabilities[:, classes.index(classname)] # get the probabilities for the current class
            class_labels = [1 if label == class_int else 0 for label in labels] # convert the labels to binary labels
            # compute the AUC for the current class -> compute it like that to be consistent with script for plotting it
            fpr, tpr, thresholds = roc_curve(class_labels, class_probs)
            if np.isnan(fpr).any() or np.isnan(tpr).any():
                continue
            roc_curves[classname] = (fpr, tpr, thresholds)
        return roc_curves


    def plot_roc_curves(self, roc_curves: dict, level: str):
        """ Plot ROC curves for each class with their corresponding AUC value.

        Args:
            roc_curves (dict): A dictionary of ROC curves for each class.
            level (str): The level of the predictions (e.g., "slide" or "patch").
        """

        # Enhanced color palette
        colors = plt.cm.tab10.colors  # 10 distinct colors from Matplotlib's tab10 colormap

        # Create the figure 
        plt.figure()

        # Plot ROC curves with enhancements
        for i, (classname, (fpr, tpr, _)) in enumerate(roc_curves.items()):
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr, 
                color=colors[i % len(colors)],  # Cycle through colors
                lw=1,                          
                label=f"{classname} (AUC = {roc_auc:.2f})"  # Add class name and AUC in legend
            )

        # Add diagonal line for random guessing
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing', lw=1)

        # Add grid for readability
        plt.grid(visible=True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)

        # Add labels and legend
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')

        # Save and display the plot (for debugging)
        save_path = os.path.join(self.plot_path, f"{level}_roc_curves_per_class.pdf")
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"ROC curves saved to: {save_path}")


    def compute_performance_metrics(self, base_name: str, level: str, performance_metrics: dict) -> dict:
        """ Compute the auc, accuracy, precision, recall, f1-score, and cross-entropy loss for the model outputs.

        Args:
            base_name (str): Base name of the file containing the model outputs and ground truths.
            level (str): The level of the predictions (e.g., "slide" or "patch").
            performance_metrics (dict): Dictionary to store the computed performance metrics.

        Returns:
            dict: A dictionary containing the computed performance metrics.
        """
        # Load the probabilities, labels, and compute the predicted labels
        probs_and_labels, model_outputs, ground_truths, predicted_labels = self.process_predictions(base_name, level)

        # compute evaluation metrics: auc, accuracy, precision, recall, f1-score and cross entropy loss
        avg_auc, _ = compute_auc(probs_and_labels, LABELS_MAP_INT_TO_STRING)
        performance_metrics[level]["auc"] = avg_auc
        performance_metrics[level]["accuracy"] = accuracy_score(ground_truths, predicted_labels)
        performance_metrics[level]["balanced_accuracy"] = balanced_accuracy_score(ground_truths, predicted_labels)

        # Compute precision (average='macro' is typically used for multi-class tasks) since 
        # micro precision = micro recall = micro F1 = accuracy, macro is the average of the scores for each class
        performance_metrics[level]["precision"] = precision_score(ground_truths, predicted_labels, average='macro', zero_division=0)
        performance_metrics[level]["recall"] = recall_score(ground_truths, predicted_labels, average='macro', zero_division=0)
        performance_metrics[level]["f1_score"] = f1_score(ground_truths, predicted_labels, average='macro', zero_division=0)

        # Compute cross-entropy loss -> make sure that # ground truth classes = # dimensions of model outputs
        relevant_model_outputs = self.get_relevant_model_outputs(model_outputs, ground_truths, predicted_labels)
        performance_metrics[level]["cross_entropy"] = log_loss(ground_truths, relevant_model_outputs)

        return performance_metrics


    def append_performance_metrics(self, base_name: str, level: str, performance_metrics: dict) -> dict:
        """ Compute the auc, accuracy, precision, recall, f1-score, and cross-entropy loss for the model outputs 
            and add them to the performance metrics dictionary.

        Args:
            base_name (str): Base name of the file containing the model outputs and ground truths.
            level (str): The level of the predictions (e.g., "slide" or "patch").                
            performance_metrics (dict): Dictionary already storing the computed performance metrics.

        Returns:
            dict: An updated dictionary containing the old plus newly computed performance metrics.
        """
        # Load the probabilities, labels, and compute the predicted labels
        probs_and_labels, model_outputs, ground_truths, predicted_labels = self.process_predictions(base_name, level)

        # compute evaluation metrics: auc, accuracy, precision, recall, f1-score and cross entropy loss
        # note that the predicted labels might contain classes that are not available in the eval data
        avg_auc, _ = compute_auc(probs_and_labels, LABELS_MAP_INT_TO_STRING)
        performance_metrics[level]["auc"].append(avg_auc)
        performance_metrics[level]["accuracy"].append(accuracy_score(ground_truths, predicted_labels))
        performance_metrics[level]["balanced_accuracy"].append(balanced_accuracy_score(ground_truths, predicted_labels))

        # Compute precision (average='macro' is typically used for multi-class tasks) since 
        # micro precision = micro recall = micro F1 = accuracy, macro is the average of the scores for each class
        performance_metrics[level]["precision"].append(precision_score(ground_truths, predicted_labels, average='macro', zero_division=0))
        performance_metrics[level]["recall"].append(recall_score(ground_truths, predicted_labels, average='macro', zero_division=0))
        performance_metrics[level]["f1_score"].append(f1_score(ground_truths, predicted_labels, average='macro', zero_division=0))

        # Compute cross-entropy loss -> make sure that # ground truth classes = # dimensions of model outputs
        relevant_model_outputs = self.get_relevant_model_outputs(model_outputs, ground_truths, predicted_labels)
        performance_metrics[level]["cross_entropy"].append(log_loss(ground_truths, relevant_model_outputs))

        return performance_metrics

    def get_relevant_model_outputs(self, model_outputs: np.ndarray, ground_truths: np.ndarray, 
                                   predicted_labels: np.ndarray) -> np.ndarray:
        """ Get the relevant model outputs for the classes that actually exist in the data.

        Args:
            model_outputs (np.ndarray): The model's probability outputs.
            ground_truths (np.ndarray): The ground truth labels.
            predicted_labels (np.ndarray): The predicted class labels.

        Returns:
            np.ndarray: The relevant model outputs.
        """
        classes_in_ground_truth, all_classes = self.check_for_unique_classes_and_labels(ground_truths, predicted_labels)

        relevant_model_outputs = model_outputs[:, classes_in_ground_truth]
        # normalize the outputs to sum to 1
        relevant_model_outputs = relevant_model_outputs / relevant_model_outputs.sum(axis=1)[:, np.newaxis]
        return relevant_model_outputs

    def save_performance_metrics(self, performance_metrics: dict, path: str):
        """ Save the computed performance metrics to a csv and a tex file and print them.

        Args:
            performance_metrics (dict): Dictionary with the computed performance metrics.
            path (str): Path to the directory where the results should be saved.
        """
        results_df = pd.DataFrame(performance_metrics)
        if self.excluded_class != "":
            results_path_prefix = os.path.join(path, f"performance_metrics_on_test_data_without_{self.excluded_class}")
        else:
            results_path_prefix = os.path.join(path, f"performance_metrics_on_test_data")
        results_df.to_csv(results_path_prefix + ".csv", index=True)
        # only save up to 4 decimal places in the latex table
        results_df.to_latex(results_path_prefix + ".tex", index=True, float_format="%.4f")
        print(f"Results\n{results_df}\nsaved to: {results_path_prefix + '.csv'}")

    def main(self):
        """ Main function for visualizing the results of the model evaluation on the test data. """
        print(f"Using resnet_dir: {self.resnet_dir}")
        ens_preds_base_path = f'{self.eval_dataset}/ensemble_predictions_and_ground_truth_labels.npz'
        sm_preds_base_path = f'{self.eval_dataset}/preds_labels_fold'
        performance_metrics = {}
        single_model_performance_metrics = {}
        single_model_std_errors = {}

        for level in ["patch", "slide"]:
            single_model_performance_metrics[level] = {"auc": [], "accuracy": [], "balanced_accuracy": [], "precision": [], "recall": [], "f1_score": [], "cross_entropy": []}

            for k in range(self.nr_of_folds):
                single_model_performance_metrics = self.append_performance_metrics(f"{sm_preds_base_path}_{k}.npz", level, single_model_performance_metrics)
            
            single_model_std_errors[f"{level}_std_error"] = {}
            for metric in single_model_performance_metrics[level].keys():
                if self.nr_of_folds != len(single_model_performance_metrics[level][metric]):
                    raise ValueError(f"Number of folds does not match the number of performance metrics for {metric}: {self.nr_of_folds} vs. {len(single_model_performance_metrics[level][metric])}")
                single_model_std_errors[f"{level}_std_error"][metric] = np.std(single_model_performance_metrics[level][metric]) / np.sqrt(self.nr_of_folds)
                single_model_performance_metrics[level][metric] = np.mean(single_model_performance_metrics[level][metric])
            performance_metrics[f"{level}_single_models"] = single_model_performance_metrics[level]
            performance_metrics[f"{level}_single_models_std_errors"] = single_model_std_errors[f"{level}_std_error"]

            performance_metrics[level] = {}
            performance_metrics = self.compute_performance_metrics(ens_preds_base_path, level, performance_metrics)

        save_path = os.path.join(self.path, self.eval_dataset)
        self.save_performance_metrics(performance_metrics, save_path)

        for level in ["patch", "slide"]:
            self.load_eval_results_and_visualize(ens_preds_base_path, level)

if __name__ == '__main__':
    args = ModelEvaluator.parse_arguments()
    evaluator = ModelEvaluator(**args)
    evaluator.main()

