"""nnU-Net trainer that writes an additional checkpoint after epoch 40."""

from os.path import join

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_STAC_400epochs_Save40(nnUNetTrainer):
    """400-epoch 2D trainer with a portable warm-up checkpoint at epoch 40."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 400

    def on_epoch_end(self):
        epoch_idx = self.current_epoch
        super().on_epoch_end()
        if (
            self.local_rank == 0
            and not self.disable_checkpointing
            and epoch_idx + 1 == 40
        ):
            current_epoch_after_super = self.current_epoch
            self.current_epoch = epoch_idx
            try:
                self.save_checkpoint(
                    join(self.output_folder, "checkpoint_epoch_0040.pth")
                )
            finally:
                self.current_epoch = current_epoch_after_super
