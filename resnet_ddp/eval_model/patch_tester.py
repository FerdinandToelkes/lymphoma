import torch
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F                                 # needed for cross entropy loss
import torchvision.transforms.v2 as T 


from torch.utils.data import DataLoader                         # needed for dataloader
from torch.nn.parallel import DistributedDataParallel as DDP    # needed for distributed testing
from torch import distributed as dist                           # needed for distributed testing


from lymphoma.diagnosis_maps import LABELS_MAP_INT_TO_STRING
from lymphoma.resnet_ddp.utils import prepare_batch, calculate_and_append_statistics
from .eval_utils import compute_class_accuracies, summarize_accuracies_to_df

class Tester:
    """
    A class for evaluating a PyTorch model using Distributed Data Parallel (DDP). This class is very similar to the Trainer class, but it is used for testing the model.

    Args:
        model (torch.nn.Module): The model to test.
        test_dataloader (DataLoader): Dataloader that loads the test data.
        patch_transform (T.Compose): Transformations to apply to the input data.
        label_transform (T.Compose): Transformations to apply to the labels.
        weights_for_loss (torch.Tensor): Weights for the loss function.
        label_smoothing (float): Label smoothing factor.
        offset (int): Offset for GPU ids -> needed when running on specific GPUs.
        output_dir (str): Output directory for saving the statistics and the model parameters.
    """
    def __init__(
        self,
        gpu_id: int,
        model: torch.nn.Module,
        test_dataloader: DataLoader,
        patch_transform: T.Compose,
        label_transform: T.Compose,
        weights_for_loss: torch.Tensor,
        label_smoothing: float,
        offset: int,                         # offset for gpu ids -> needed when running on specific gpus
        output_dir: str,
    ) -> None:
        self.gpu_id = gpu_id
        self.offset = offset
        self.model = model
        self.test_dataloader = test_dataloader 
        self.patch_transform = patch_transform
        self.label_transform = label_transform
        self.weights_for_loss = weights_for_loss
        self.label_smoothing = label_smoothing
        self.statistics = {'loss': [], 'accuracy': []}
        self.output_dir = output_dir
        # wrap model in DDP
        self.model = DDP(self.model, device_ids=[self.gpu_id]) 
        self.model.eval()
        # setup labels map
        self.labels_map = LABELS_MAP_INT_TO_STRING


    ##########################################################################################################################
    ################################################### MAIN FUNCTION ########################################################
    ##########################################################################################################################

    def run_testing(self):
        """ Computes the general test statistics and the accuracy for each class. Augmented test data is expected. 
        
        Returns:
            dict: Dictionary containing the class accuracies, the confusion matrix, the loss and the total accuracy.
        """
        # setup as tensors on gpu -> needed for distributed testing
        self.loss = torch.tensor(0.0).to(self.gpu_id)
        self.correct_preds = torch.tensor(0).to(self.gpu_id)
        total_len_test_data = torch.tensor(0).to(self.gpu_id)
    
        # prepare to count predictions for each class
        self.classes = tuple(self.labels_map.values())
        self.correct_pred_per_class = {classname: torch.tensor(0).to(self.gpu_id) for classname in self.classes}
        self.total_occurances = {classname: torch.tensor(0).to(self.gpu_id) for classname in self.classes}
        self.confusion_matrix =  {classname: {classname: torch.tensor(0).to(self.gpu_id) for classname in self.classes} for classname in self.classes}

        # iterate over data without computing gradients
        print(f"[GPU{self.gpu_id}] Start evaluation | Steps: {self.test_dataloader.__len__()}")
        with torch.no_grad():
            for x, y in self.test_dataloader:
                # update total length of test data that model has seen
                total_len_test_data += len(y)
                x, y = prepare_batch(x, y, self.patch_transform, self.label_transform, self.gpu_id)
                # Run testing for current batch
                self._run_test_batch(x, y)

        # compute the test loss and accuracy (also for each class) -> wait for all gpus to finish
        dist.barrier()
        calculate_and_append_statistics(
            self.loss, self.correct_preds, total_len_test_data, 'loss', 
            'accuracy', 0, "Testing", self.gpu_id, self.statistics
        )
        class_accuracies = compute_class_accuracies(self.correct_pred_per_class, self.total_occurances, self.classes)
        np_confusion_matrix = self.reduce_confusion_matrix_and_convert_to_np()
        # finish testing
        if self.gpu_id == self.offset:
            # return the class accuracies and the confusion matrix
            df_class_accuracies, total_acc = summarize_accuracies_to_df(class_accuracies)
            return_dict = {
                "df_class_accuracies": df_class_accuracies,
                "np_confusion_matrix": np_confusion_matrix,
                "total_loss": self.statistics['loss'][-1],
                "total_acc": total_acc
            }
            return return_dict
        return None
            
      

    ##########################################################################################################################
    ######################################## helper functions for evaluate() #################################################
    ##########################################################################################################################
   
    def _run_test_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        """Runs one batch of test data.
        
        Args:
            inputs (torch.Tensor): The input data.
            labels (torch.Tensor): The labels.
        """
        # only forward
        outputs = self.model(inputs) 
        # get the index of the class with the highest probability (using that softmax is monotonically increasing (*))
        _, predictions = torch.max(outputs, 1) 
        loss = F.cross_entropy(outputs, labels, weight=self.weights_for_loss, label_smoothing=self.label_smoothing) # this computes the softmax internally (*)

        # collect the correct predictions for each class for current batch
        for label, prediction in zip(labels, predictions):
            # add the prediction made for the current label to class it belongs to
            self.confusion_matrix[self.classes[label]][self.classes[prediction]] += 1
            self.total_occurances[self.classes[label]] += 1
            if label == prediction:
                self.correct_pred_per_class[self.classes[label]] += 1

        # update the total statistics (i.e. averaged over all classes)
        self.loss += loss.item() * inputs.size(0) # inputs.size(0) is the batch size and loss.item() is the average loss of the batch
        self.correct_preds += torch.sum(predictions == labels)

    
    def reduce_confusion_matrix_and_convert_to_np(self):
        """ Reduces the confusion matrix by summing over all gpus converts the tensors to integers 
            and returns the numpy array of the confusion matrix where the rows represent the actual 
            labels and the columns represent the predicted labels.
        
        Returns:
            np.ndarray: The confusion matrix as a numpy array.
        """
        for actual_label in self.classes:
            for predicted_label in self.classes:
                dist.all_reduce(self.confusion_matrix[actual_label][predicted_label], op=dist.ReduceOp.SUM)
                self.confusion_matrix[actual_label][predicted_label] = int(self.confusion_matrix[actual_label][predicted_label].cpu().item())

        # obtain the confusion matrix as a numpy array of integers
        np_confusion_matrix = np.zeros((len(self.classes), len(self.classes)))
        for i, actual_label in enumerate(self.classes):
            for j, predicted_label in enumerate(self.classes):
                # convert the tensor to a float and save it in the confusion matrix
                np_confusion_matrix[i, j] = self.confusion_matrix[actual_label][predicted_label]
        return np_confusion_matrix

        
    def save_statistics(self):
        """ Saves the loss and performance statistics to a CSV file in a seperate output directory. """
        df = pd.DataFrame(self.statistics)
        path = os.path.join(self.output_dir, "patch_test_statistics.csv")
        print(f"[GPU{self.gpu_id}] Saving statistics to {path}")
        df.to_csv(path, index=False)


    def print_statistics(self, class_accuracies: dict):
        """ Prints the final test loss, accuracy and class accuracies to the console

        Args:
            class_accuracies (dict): Dictionary containing the class accuracies.
        """
        print(f"[GPU{self.gpu_id}] | Test loss: {self.statistics['loss'][-1]}")
        print(f"[GPU{self.gpu_id}] | Test accuracy: {self.statistics['accuracy'][-1]}")
        print(f"[GPU{self.gpu_id}] | Class accuracies:")
        for classname, stats in class_accuracies.items():
            print(f"{classname}: {stats['corrects']}/{stats['total']} = {stats['accuracy']}%")

    





    

                

            
                

            