import torch
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F                                 # needed for cross entropy loss
import torchvision.transforms.v2 as T 


from torch.utils.data import DataLoader, Dataset                # needed for dataloader
from torch.nn.parallel import DistributedDataParallel as DDP    # needed for distributed testing
from torch import distributed as dist                           # needed for distributed testing
from torch.utils.data.distributed import DistributedSampler     # needed for distributed testing

from pamly import Diagnosis         

from lymphoma.diagnosis_maps import LABELS_MAP_INT_TO_STRING                         
from .eval_utils import compute_class_accuracies, summarize_accuracies_to_df, compute_auc
from lymphoma.resnet_ddp.utils import prepare_batch

# to keep myself from changing it -> has to be one for testing on slide level
BATCH_SIZE = 1


class PatchesTestDataset(Dataset):
    """ Class to load the patches dataset for especially for testing the model on slide level. 
        It used is later used to load slide by slide all patches."""

    def __init__(self, patches_list: list, img_dir: str):
        # make sure the patches_list is sorted (patch_0_coords..., patch_1_coords..., ...)
        patches_list = sorted(patches_list, key=lambda x: int(x.split("_")[1]))
        self.patches_list = patches_list
        self.img_dir = img_dir

    def __len__(self):
        return len(self.patches_list)

    def __getitem__(self, idx) -> tuple:
        """ Loads the patch and its label. 
        
        Args:
            idx (int): Index of the patch to load.  

        Returns:
            tuple: The patch and its label.
        """
        # load the patch and its label
        patch_path = os.path.join(self.img_dir, self.patches_list[idx]) # 0th column is the file name
        patch_with_label = torch.load(patch_path) 
        patch = patch_with_label[0] 
        label = patch_with_label[1]
        return patch, label



class Tester():
    """
    A class for evaluating a PyTorch model using Distributed Data Parallel (DDP).

    Args:
        gpu_id (int): The GPU id to use for testing.
        model (torch.nn.Module): The model to test.
        test_slide_df (pd.DataFrame): A dataframe containing the test slides and their labels.
        patch_transform (T.Compose): A composition of transforms to apply to the patches.
        label_transform (T.Compose): A composition of transforms to apply to the labels.
        tumor_probability_threshold (int): Threshold for tumor probability of a patch to be considered as relevant.
        offset (int): Offset for GPU ids -> needed when running on specific GPUs.
        output_dir (str): Output directory for saving the statistics and the model parameters.
        do_final_pred_with_probs (bool): Whether to do the final prediction with probabilities.
    """
    def __init__(
        self,
        gpu_id: int,
        model: torch.nn.Module,
        test_slide_df: pd.DataFrame,
        patch_transform: T.Compose,
        label_transform: T.Compose,
        tumor_probability_threshold: int,                    # threshold for tumor probability of a patch to be considered as relevant
        offset: int,                                         # offset for gpu ids -> needed when running on specific gpus
        output_dir: str,
        do_final_pred_with_probs: bool = False,
    ) -> None:
        self.gpu_id = gpu_id
        self.tumor_probability_threshold = tumor_probability_threshold
        self.offset = offset
        self.model = model
        self.test_slide_df = test_slide_df
        self.patch_transform = patch_transform
        self.label_transform = label_transform
        self.output_dir = output_dir
        # wrap model in DDP and set to eval mode
        self.model = DDP(self.model, device_ids=[self.gpu_id]) 
        self.model.eval()
        
        self.labels_map_int_to_string = LABELS_MAP_INT_TO_STRING
        self.do_final_pred_with_probs = do_final_pred_with_probs


    ##########################################################################################################################
    ################################################### MAIN FUNCTION ########################################################
    ##########################################################################################################################

    # note: no calculate_and_append_statistics needed here since we are computing the auc seperately
    def run_testing(self, data_mount_dir: str, data_dir: str, data_specifier: str) -> dict:
        """ Runs the testing procedure. 
        
        Args:
            data_mount_dir (str): The directory containing the data.
            data_dir (str): The directory containing the slides.
            data_specifier (str): The specifier for the data directory.

        Returns:
            dict: A dictionary containing the statistics and the performance metrics.
        """
        # setup to track statistics and to count predictions for each class on slide level; will be summed over all gpus
        self.classes = tuple(self.labels_map_int_to_string.values()) # get the class names
        self.correct_pred = {classname: torch.tensor(0).to(self.gpu_id) for classname in self.classes}
        self.total_occurances = {classname: torch.tensor(0).to(self.gpu_id) for classname in self.classes}
        self.wrong_classified_slides = {"slide_name": [], "predicted_class": [], "actual_class": []}
        # containers needed for computing various performance metrics
        self.probs_and_labels_for_all_slides = {"probs": [], "labels": []}
        self.probs_and_labels_for_all_patches = {"probs": [], "labels": []}
            
        # iterate over all test slides which are stored in the test_slide_df containing also the dataset name
        nr_slides = len(self.test_slide_df.index)
        for i, (row_idx, row) in enumerate(self.test_slide_df.iterrows()):
            slide_name = row["filename"]
            dataset = row["dataset"]
            if self.gpu_id == self.offset:
                print(f"[GPU{self.gpu_id}] Testing slide {i+1}/{nr_slides}: {slide_name} of {dataset}            ", end="\r")
            # count number of slides for each class giving the total number of occurences for each class
            label = slide_name.split("/")[0]
            self.total_occurances[label] += 1
            self.probs_and_labels_for_all_slides["labels"].append(int(Diagnosis(label)))
            # get the patch names for current slide
            path_to_slide = os.path.join(data_mount_dir, dataset, data_specifier, data_dir, slide_name)
            patch_names = os.listdir(path_to_slide)
            patch_names = [patch_name for patch_name in patch_names if patch_name.startswith("patch") and patch_name.endswith(".pt")]
            if self.tumor_probability_threshold != 0:
                patch_names = [patch_name for patch_name in patch_names \
                               if int(patch_name.split("tp=")[-1].split(".")[0]) >= self.tumor_probability_threshold]
            if len(patch_names) == 0:
                print(f"[GPU{self.gpu_id}] No patches found for slide {slide_name}.")
                continue
    
            # create dataset and dataloader for each slide 
            one_slide_dataset = PatchesTestDataset(patch_names, path_to_slide)
            one_slide_dataloader = DataLoader(one_slide_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                              sampler=DistributedSampler(one_slide_dataset, shuffle=False), 
                                              drop_last=True, num_workers=1, prefetch_factor=10)

            # run testing for each slide
            self.run_testing_for_one_slide(one_slide_dataloader, slide_name)
            self.probs_and_labels_for_all_patches["labels"].extend([int(Diagnosis(label))] * len(one_slide_dataloader))

            
        # compute the test loss and accuracy (also for each class) -> wait for all gpus to finish
        dist.barrier()
        class_accuracies = compute_class_accuracies(self.correct_pred, self.total_occurances, self.classes, eval_type="Slide")
        # finish testing
        if self.gpu_id == self.offset:
            # get the relevant statistics and accuracies 
            df_class_accuracies, total_acc = summarize_accuracies_to_df(class_accuracies)
            total_auc, auc_values = compute_auc(self.probs_and_labels_for_all_slides, self.labels_map_int_to_string)
            # these have been averaged over all gpus
            return_dict = {
                "df_class_accuracies": df_class_accuracies,
                "wrong_classified_slides": self.wrong_classified_slides,
                "total_acc": total_acc,
                "total_auc": total_auc,
                "auc_values": auc_values,
                "probs_and_labels_for_all_slides": self.probs_and_labels_for_all_slides,
                "probs_and_labels_for_all_patches": self.probs_and_labels_for_all_patches,
            }
            return return_dict
        return None


    def run_testing_for_one_slide(self, dataloader: DataLoader, slide_name: str):
        """ Computes the general test statistics and the accuracy for each class. 
            Augmented test data is expected. 
        
        Args:
            dataloader (DataLoader): The dataloader for the patches of the slide.
            slide_name (str): The name of the slide.
        """
        self.predictions_for_current_slide = {classname: torch.tensor(0).to(self.gpu_id) for classname in self.classes}
        self.probabilities_for_current_slide = torch.zeros(len(self.classes)).to(self.gpu_id)
        
        # iterate over all data without computing gradients
        with torch.no_grad():
            for x, y in dataloader:
                x, y = prepare_batch(x, y, self.patch_transform, self.label_transform, self.gpu_id)
                # Run testing for current batch
                self._run_test_batch(x)
                # save label of slide, remember that the labels are the same for all patches of a slide
                ground_truth_of_slide = y[0].cpu().item()

        # sum the predictions for current slide over all gpus
        dist.barrier()
        nr_of_gpus = dist.get_world_size()
        for key in self.predictions_for_current_slide.keys():
            dist.all_reduce(self.predictions_for_current_slide[key], op=dist.ReduceOp.SUM)
        for i in range(len(self.probabilities_for_current_slide)):
            dist.all_reduce(self.probabilities_for_current_slide[i], op=dist.ReduceOp.SUM)
            norm_probs_for_current_slide = self.probabilities_for_current_slide / (len(dataloader) * nr_of_gpus * BATCH_SIZE)
        # finish testing for current slide
        if self.gpu_id == self.offset:
            # perform majority voting for final diagnosis of current slide -> get class with most predictions, (.get -> apply max on values of dict)
            final_prediction = max(self.predictions_for_current_slide, key=self.predictions_for_current_slide.get)
            # get index of class with highest probability
            final_pred_with_probs = torch.argmax(self.probabilities_for_current_slide)

            # convert to int for comparison with ground truth
            final_prediction = int(Diagnosis(final_prediction)) 
            final_pred_with_probs = int(final_pred_with_probs)
            final_prediction = final_pred_with_probs if self.do_final_pred_with_probs else final_prediction

            if final_prediction == ground_truth_of_slide:
                self.correct_pred[self.labels_map_int_to_string[int(final_prediction)]] += 1
            else:
                # save wrongly classified slides
                self.wrong_classified_slides['slide_name'].append(slide_name)
                self.wrong_classified_slides['predicted_class'].append(self.labels_map_int_to_string[int(final_prediction)])
                self.wrong_classified_slides['actual_class'].append(self.labels_map_int_to_string[int(ground_truth_of_slide)])
            # update probabilities
            self.probs_and_labels_for_all_slides["probs"].append(norm_probs_for_current_slide.cpu().numpy())

    

    ##########################################################################################################################
    ######################################## helper functions for evaluate() #################################################
    ##########################################################################################################################

   
    def _run_test_batch(self, inputs: torch.Tensor):
        """ Runs one batch of test data.

        Args:
            inputs (torch.Tensor): The input data. 
        """
        # only forward
        outputs = self.model(inputs) 
        # softmax to get probabilities
        outputs = F.softmax(outputs, dim=1)
        # get the index of the class with the highest probability (using that softmax is monotonically increasing)
        _, predictions = torch.max(outputs, 1) # 1 is the dimension along which to take the max

        # collect the predictions for each class for current batch
        for i, prediction in enumerate(predictions):
            self.predictions_for_current_slide[self.labels_map_int_to_string[int(prediction)]] += 1

        # collect the probabilities for each class for current
        for i, prob in enumerate(outputs[0]): # since batch size is 1
            self.probabilities_for_current_slide[i] += prob
            self.probs_and_labels_for_all_patches["probs"].append(prob.cpu().numpy())
    
    def save_wrong_classified_slides(self):
        """ Saves the wrongly classified slides to a CSV file in a seperate output directory. """
        # convert the wrongly classified slides to a pandas dataframe
        df = pd.DataFrame(self.wrong_classified_slides)
        path = os.path.join(self.output_dir, "wrongly_classified_slides.csv")
        print(f"[GPU{self.gpu_id}] Saving wrongly classified slides to {path}")
        df.to_csv(path, index=False)


    def print_statistics(self, class_accuracies: dict):
        """ Prints the statistics to the console.

        Args:
            class_accuracies (dict): The class accuracies.
        """
        print(f"[GPU{self.gpu_id}] | Class accuracies:")
        total_slides = 0
        total_corrects = 0
        for classname, stats in class_accuracies.items():
            total_slides += stats['total']
            total_corrects += stats['corrects']
            print(f"{classname}: {stats['corrects']}/{stats['total']} = {stats['accuracy']}%")
        print(f"Total accuracy: {total_corrects}/{total_slides} = {100 * total_corrects / total_slides:.2f}%")
