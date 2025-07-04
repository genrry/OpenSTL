from .base_method import Base_method


class DYffusion(Base_method):
    r"""DYffusion - Dynamic Diffusion Model

    Implementation of training and validation logic for Dyffusion model.
    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        from openstl.models import DYffusion_Model

        return DYffusion_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        """
        Forward pass for video prediction using DYffusion.

        Args:
            batch_x: Input frames [B, T_in, C, H, W]
            batch_y: Target frames [B, T_out, C, H, W] (optional, used during training)

        Returns:
            pred_y: Predicted frames [B, T_out, C, H, W]
        """
        raise NotImplementedError(
            "DYffusion does not support direct forward pass with batch_y. "
            "Use training_step or validation_step for training and inference."
        )
        aft_seq_length = self.hparams.aft_seq_length

        if self.training and batch_y is not None:
            # Training: use DYffusion p_losses
            loss_dict = self.model.forward_training(batch_x, batch_y)
            # Return the final prediction for loss computation
            # In DYffusion training, we don't need explicit predictions, just losses
            return batch_y  # Return target for compatibility with base_method loss
        else:
            # Inference: generate future frames using DYffusion sampling
            pred_y = self.model.forward_inference(batch_x, aft_seq_length)
            return pred_y

    def training_step(self, batch, batch_idx):
        """
        Training step using DYffusion's dual-loss training procedure.
        """
        batch_x, batch_y = batch

        # Get DYffusion training losses
        loss_dict = self.model.forward_training(batch_x, batch_y)

        # Extract losses
        total_loss = loss_dict["loss"]
        loss_forward = loss_dict["loss_forward"]
        loss_forward2 = loss_dict["loss_forward2"]

        # Log losses
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_forward", loss_forward, on_step=True, on_epoch=True)
        self.log("train_loss_forward2", loss_forward2, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step - can use either training losses or inference MSE.
        """
        batch_x, batch_y = batch

        if (
            hasattr(self.hparams, "use_inference_for_validation")
            and self.hparams.use_inference_for_validation
        ):
            # Use inference for validation
            pred_y = self.model.forward_inference(batch_x, self.hparams.aft_seq_length)
            loss = self.criterion(pred_y, batch_y)
        else:
            # Use training losses for validation
            loss_dict = self.model.forward_training(batch_x, batch_y)
            loss = loss_dict["loss"]

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss
