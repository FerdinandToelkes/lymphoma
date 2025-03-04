import torch
import os
import pandas as pd                                             # needed for statistics
import torch.nn.functional as F                                 # needed for cross entropy loss
import time                                                     # needed for time measurement of epochs
import torchvision.transforms.v2 as T 


from torch.utils.data import DataLoader                         # needed for dataloader
from torch.nn.parallel import DistributedDataParallel as DDP    # needed for distributed training
from torch import distributed as dist                           # needed for distributed training

from lymphoma.resnet_ddp.eval_model.patch_tester import Tester as PatchTester
from lymphoma.resnet_ddp.eval_model.slide_tester import Tester as SlideTester
from lymphoma.resnet_ddp.utils import prepare_batch, calculate_and_append_statistics


class Trainer:
    """
    A class for training a PyTorch model using Distributed Data Parallel (DDP).

    Args:
        gpu_id (int): The ID of the GPU to use.
        model (torch.nn.Module): The PyTorch model to train.
        train_dataloader (DataLoader): The DataLoader for the training data.
        val_dataloader (DataLoader): The DataLoader for the validation data.
        val_slide_df (any): The dataframe containing the slide names and labels for slide validation.
        patch_transform (torchvision.transforms.Compose): The transform to apply to the patches.
        label_transform (torchvision.transforms.Compose): The transform to apply to the labels.
        max_epochs (int): The maximum number of epochs to train for.
        warmup_epochs (int): The number of epochs to run the warmup scheduler.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        warmup_scheduler (torch.optim.lr_scheduler): The warmup scheduler to use for training.
        main_scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use for training.
        save_every (int): The number of epochs between saving the model.
        save_model (bool): Whether to save the model or not.
        weights_for_loss (torch.Tensor): The weights for the weighted cross entropy loss.
        label_smoothing (float): The label smoothing factor for the cross entropy loss.
        patience (int): The number of epochs to wait before stopping training if no improvement in validation accuracy.
        offset (int): The offset for GPU IDs, needed when running on specific GPUs.
        output_dir (str): The path to the directory to save the model to.
        data_mount_dir (str): The path to the directory containing the data.
        data_dir (str): The path to the directory containing the data.
        data_specifier (str): The specifier for the data.
        tumor_probability_threshold (float): The threshold for the tumor probability for slide validation.
        statistics_path (str): The path to the CSV file to save loss and performance statistics.
        snapshot_path (str): The path to the snapshot file to resume training from.
    """
    def __init__(
        self,
        gpu_id: int,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader, 
        val_slide_df: any,             # either slide dataframe or none
        patch_transform: T.Compose,
        label_transform: T.Compose,
        max_epochs: int,
        warmup_epochs: int,
        optimizer: torch.optim.Optimizer,
        warmup_scheduler: torch.optim.lr_scheduler,
        main_scheduler: torch.optim.lr_scheduler,
        save_every: int, 
        save_model: bool,                       # whether to save the model or not (during cv saving not needed)
        weights_for_loss: torch.Tensor,         # weights for weighted cross entropy loss
        label_smoothing: float,                 # label smoothing factor for cross entropy loss
        patience: int,                          # number of epochs to wait before stopping training if no improvement in validation accuracy
        offset: int,                            # offset for gpu ids -> needed when running on specific gpus
        output_dir: str,
        data_mount_dir: str,
        data_dir: str,                          # only needed for slide validation, otherwise already used in val_dataloader
        data_specifier: str,
        tumor_probability_threshold: float,     # only needed for slide validation, otherwise already used in val_dataloader
        statistics_path: str,
        snapshot_path: str = "snapshot.pt"
    ) -> None:
        self.gpu_id = gpu_id
        self.offset = offset
        self.model = model
        self.train_dataloader = train_dataloader 
        self.val_dataloader = val_dataloader            # either patch dataloader or slide_names is none
        self.val_slide_df = val_slide_df             # either slide dataframe or none
        self.patch_transform = patch_transform
        self.label_transform = label_transform
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.save_every = save_every
        self.save_model = save_model
        self.weights_for_loss = weights_for_loss
        self.label_smoothing = label_smoothing
        self.epochs_run = 0
        self.patience = patience
        self.val_metric = 'loss' if self.val_slide_df is None else 'auc'
        self.statistics = {'train_loss': [], 'train_acc': [], f'val_{self.val_metric}': [], 'val_acc': []}
        self.output_dir = output_dir
        self.data_mount_dir = data_mount_dir
        self.data_dir = data_dir
        self.data_specifier = data_specifier
        self.tumor_probability_threshold = tumor_probability_threshold
        self.snapshot_path = os.path.join(output_dir, snapshot_path) 
        self.statistics_path = os.path.join(output_dir, statistics_path)
        # load snapshot and statistics if they exist in current directory -> they later will be saved in output directory!
        if os.path.exists(snapshot_path):
            print(f"Resuming training from snapshot at {snapshot_path} and loss statistics at {statistics_path}")
            self._load_snapshot(snapshot_path)
            self._load_statistics_as_dict(statistics_path)
        # wrap model in DDP
        self.model = DDP(self.model, device_ids=[self.gpu_id]) 

    ##########################################################################################################################
    ########################################## helper functions needed in init ###############################################
    ##########################################################################################################################

    
    def _load_snapshot(self, snapshot_path: str):
        """
        Loads a snapshot of a previous training run.

        Args:
            snapshot_path (str): Path to the snapshot.
        """
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    
    def _load_statistics_as_dict(self, statistics_path: str):
        """
        Loads the loss and performance statistics of a previous run from a CSV file.

        Args:
            statistics_path (str): Path to the CSV file.
        """
        df = pd.read_csv(statistics_path)
        self.statistics = df.to_dict(orient='list')
        print(f"Resuming training from statistics at Epoch {self.epochs_run}")

    ##########################################################################################################################
    ################################################## main functions ########################################################
    ##########################################################################################################################

    def train(self, validate_every: int, path_to_best_model: str = "best_model.pt") -> dict:
        """
        Trains the model for max_epochs epochs.

        Args:
            validate_every (int): Number of epochs between running the validation.
            path_to_best_model (str, optional): Path to save the best model parameters. Defaults to "best_model.pt".

        Returns:
            Dictionary containing the best accuracy and loss or None if not on the 'root' gpu.
        """
        # start with best loss as infinity if we are running on patch data and as -infinity if we are running on slide data
        # -> either minimize loss or maximize auc
        self.best_val_metric = float('inf') if self.val_metric == 'loss' else float('-inf')
        self.early_stopping_count = 0

        for epoch in range(self.epochs_run, self.max_epochs):
            # remeber offset when we are running for example only on gpus 6 and 7
            self._run_epoch(epoch, path_to_best_model, validate_every) 
            # wait for all gpus to finish the epoch
            dist.barrier() 
            early_stop = self.determine_early_stop()

            if early_stop == 1:
                if self.gpu_id == self.offset:
                    print(f"self.early_stopping_count: {self.early_stopping_count}")
                    print(f"self.patience: {self.patience}")
                    print(f"Early stopping at epoch {epoch}")
                break
            if self.gpu_id == self.offset and epoch % self.save_every == 0:
                # save snapshot
                self._save_model(epoch, self.snapshot_path)

        if self.gpu_id == self.offset:
            # remove the last snapshot if a snapshot even exists such that it won't be loaded when restarting training
            if os.path.exists(self.snapshot_path):
                print(f"Removing snapshot at {self.snapshot_path}")
                os.remove(self.snapshot_path)
            # save training statistics and return the best accuracy and loss
            if self.save_model:
                self._save_statistics()
            best_acc = max(self.statistics['val_acc'])
            best_loss = max(self.statistics[f'val_{self.val_metric}']) if self.val_metric == 'auc' else min(self.statistics[f'val_{self.val_metric}'])
            return {"best_acc": best_acc, "best_loss": best_loss}
        
        # return None if not on the 'root' gpu
        return None
            
    def determine_early_stop(self) -> int:
        """ Determines whether to apply early stopping or not. 
        
        Returns:
            early_stop (int): 1 if early stopping should be applied, 0 otherwise.
        """
        if self.gpu_id == self.offset:
            if self.early_stopping_count > self.patience:
                early_stop = torch.tensor(1, device=self.gpu_id)  # Signal to stop
            else:
                early_stop = torch.tensor(0, device=self.gpu_id)  # Signal to continue
        else:
            early_stop = torch.tensor(0, device=self.gpu_id)  # Placeholder for other GPUs

        # Broadcast the early stopping decision from the root GPU to all other GPUs
        dist.broadcast(early_stop, src=self.offset)
        
        # Convert the early_stop signal to a Python integer for clarity
        early_stop = early_stop.item()
        return early_stop

    def _run_epoch(self, epoch: int, path_to_best_model: str, validate_every: int) -> bool: 
        """
        Runs the training and validation for one epoch.

        Args:
            epoch (int): Number of epochs that have been run.
            path_to_best_model (str): Path to save the best model parameters.
            validate_every (int, optional): Number of epochs between running the validation. 

        Returns:
            running_status (bool): True if the training should continue, False if early stopping should be applied.
        """
        # setup statistics for training and validation
        self.training_loss, self.validation_loss = torch.tensor(0.0).to(self.gpu_id), torch.tensor(0.0).to(self.gpu_id)
        self.training_corrects, self.validation_corrects = torch.tensor(0).to(self.gpu_id), torch.tensor(0).to(self.gpu_id)
        total_len_train_data, total_len_validation_data = torch.tensor(0).to(self.gpu_id), torch.tensor(0).to(self.gpu_id)

        # run one epoch of training data
        self._run_training_epoch(epoch, total_len_train_data)
        # wait for all gpus to finish the epoch
        dist.barrier() 
        # Run validation every nth epoch and at the last epoch
        if epoch % validate_every == 0 or epoch == self.max_epochs - 1:
            self.model.eval()
            # either run patch or slide validation
            if self.val_dataloader is not None:
                epoch_val_metric = self._run_patch_validation_epoch(epoch)
            else:
                epoch_val_metric = self._run_slide_validation_epoch(epoch)
            self._update_best_loss_and_early_stopping(epoch_val_metric, path_to_best_model, epoch)
            
        # append nan to validation statistics if no validation has been run
        else:
            self.statistics[f'val_{self.val_metric}'].append(float('nan'))
            self.statistics['val_acc'].append(float('nan'))

    ##########################################################################################################################
    ######################################### helper functions for train() ###################################################
    ##########################################################################################################################

    def _run_training_epoch(self, epoch: int, total_len_train_data: torch.Tensor) -> None:
        """ Runs the training for one epoch. 
        
        Args:
            epoch (int): Number of epochs that have been run.
            total_len_train_data (torch.Tensor): The total length of the training data that the model has seen.
        """
        if self.gpu_id == self.offset:
            print(f"GPU {self.gpu_id} self.early_stopping_count: {self.early_stopping_count}")
            print(f"[GPU{self.gpu_id}] Start epoch {epoch} | Steps: {self.train_dataloader.__len__()}")
        # setup
        start_epoch = time.time()
        # set epoch for correct shuffling of data
        self.train_dataloader.sampler.set_epoch(epoch)
        self.model.train()
        # iterate over training samples
        for x, y in self.train_dataloader:
            # update total length of training data that model has seen
            total_len_train_data += len(y)
            x, y = prepare_batch(x, y, self.patch_transform, self.label_transform, self.gpu_id)
            # Run training for current batch
            self._run_trainings_batch(x, y)

        # Update the scheduler and print time for training epoch
        if epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()
        end_epoch = time.time()
        if self.gpu_id == self.offset:
            print(f"[GPU{self.gpu_id}] Time for training epoch: {end_epoch - start_epoch}s")
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Learning rate after epoch = {current_lr:.6f}\n")

        # Calculate and save the train and validation statistics
        dist.barrier() # wait for all gpus to finish the epoch
        calculate_and_append_statistics(
            self.training_loss, self.training_corrects, total_len_train_data,
            'train_loss', 'train_acc', epoch, 'Training', self.gpu_id, self.statistics
            )
        
    
    def _run_patch_validation_epoch(self, epoch: int) -> torch.Tensor:
        """ Runs the validation for one epoch. 
        
        Args:
            epoch (int): Number of epochs that have been run.

        Returns:
            patch_loss (torch.Tensor): The loss of the patch
        """
        if self.gpu_id == self.offset:
            print(f"[GPU{self.gpu_id}] Start patch validation epoch {epoch} | Steps: {self.val_dataloader.__len__()}")
        # setup for validation on patch level
        tester = PatchTester(self.gpu_id, self.model, self.val_dataloader, self.patch_transform, self.label_transform, self.weights_for_loss, self.label_smoothing, self.offset, self.output_dir)
        # run validation on patch level, results dict is none if not on the 'root' gpu
        results = tester.run_testing()
        if self.gpu_id == self.offset:
            patch_loss = results['total_loss']
            patch_acc = results['total_acc']
            print(f"Patch val loss: {patch_loss}")
            print(f"Patch val accuracy: {patch_acc}")
            # append patch accuracy and loss to statistics
            self.statistics[f'val_{self.val_metric}'].append(patch_loss)
            self.statistics['val_acc'].append(patch_acc)
            return patch_loss
        return None

    
    def _run_slide_validation_epoch(self, epoch: int) -> torch.Tensor:
        """ Runs the validation for one epoch. 
        
        Args:
            epoch (int): Number of epochs that have been run.

        Returns:
            slide_auc (torch.Tensor): The auc of the slide
        """
        if self.gpu_id == self.offset:
            print(f"[GPU{self.gpu_id}] Start slide validation epoch {epoch}")
        # setup and run validation on slide level
        tester = SlideTester(self.gpu_id, self.model, self.val_slide_df, self.patch_transform, self.label_transform,
                            self.tumor_probability_threshold, self.offset, self.output_dir)
        # run validation on slide level, results dict is none if not on the 'root' gpu
        results = tester.run_testing(self.data_mount_dir, self.data_dir, self.data_specifier)
        if self.gpu_id == self.offset:
            slide_auc = results['total_auc']
            slide_acc = results['total_acc']
            print(f"Slide val AUC: {slide_auc}")
            print(f"Slide val Accuracy: {slide_acc}")
            # append slide accuracy and auc to statistics
            self.statistics[f'val_{self.val_metric}'].append(slide_auc)
            self.statistics['val_acc'].append(slide_acc)
            return slide_auc

        return None

        
    
    def _update_best_loss_and_early_stopping(self, epoch_val_metric: torch.Tensor, path_to_best_model: str, 
                                             epoch: int) -> None:
        """ Updates the best loss metric and early stopping count. 
        
        Args:
            epoch_val_metric (torch.Tensor): The loss of the patch validation or the auc of the slide validation.
            path_to_best_model (str): Path to save the best model parameters.
            epoch (int): Number of epochs that have been run.
        """
        # deep copy the model if it is the best one so far and we are on the 'root' gpu (with id offset)
        if self.gpu_id == self.offset: 
            # either minimize loss or maximize auc
            cond = epoch_val_metric > self.best_val_metric if self.val_metric == 'auc' else epoch_val_metric < self.best_val_metric
            if cond:
                self.best_val_metric = epoch_val_metric     
                if self.save_model:           
                    path = os.path.join(self.output_dir, path_to_best_model)
                    self._save_model(epoch, path)
                # reset early stopping count to 0
                self.early_stopping_count = 0
            else:
                self.early_stopping_count += 1
    

    def _run_trainings_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        """ Runs the training for one batch.

        Args:
            inputs (torch.Tensor): Input tensor.
            labels (torch.Tensor): Label tensor.
        """
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # forward + backward + optimize
        with torch.set_grad_enabled(True):  # Should not be needed after model.train()
            outputs = self.model(inputs) # this has shape (batch_size, num_classes) since we have a linear layer at the end
            _, preds = torch.max(outputs, 1) # get the index of the max log-probability
            loss = F.cross_entropy(outputs, labels, weight=self.weights_for_loss, label_smoothing=self.label_smoothing) # this contains a log_softmax() call
            loss.backward()
            self.optimizer.step()

            # update statistics
            self.training_loss += loss.item() * inputs.size(0) # inputs.size(0) is the batch size and loss.item() is the average loss of the batch
            self.training_corrects += torch.sum(preds == labels) 

    
        
    def _save_model(self, epoch: int, path: str):
        """
        Saves a snapshot of the current model state and the number of epochs that have been run.

        Args:
            epoch (int): Number of epochs that have been run.
            path (str, optional): Path to save the model. 
        """
        snapshot = {}  
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, path)
        print(f"Epoch {epoch} | Training snapshot saved at {path}")

    def _save_statistics(self):
        """ Saves the loss and performance statistics to a CSV file in a seperate output directory. """
        df = pd.DataFrame(self.statistics)
        print(f"Saving statistics at {self.statistics_path}")
        df.to_csv(self.statistics_path, index=False)


    



