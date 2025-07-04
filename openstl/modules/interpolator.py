"""
Interpolator module for DYffusion implementation in OpenSTL.
This module provides the interpolation functionality required by DYffusion.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from contextlib import contextmanager


class Interpolator(nn.Module):
    """
    Base interpolation class that mimics the original DYffusion interpolator.
    This is a modified version for OpenSTL integration.
    """

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 64,
        window: int = 1,
        horizon: int = 10,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.channels = channels
        self.hidden_dim = hidden_dim
        self.window = window
        self.horizon = horizon
        self.true_horizon = horizon
        self.dropout_rate = dropout_rate

        # Build the interpolator network
        self.encoder = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        self.time_proj = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, channels, 3, padding=1),
        )

        self.dropout = nn.Dropout2d(dropout_rate)

    @contextmanager
    def inference_dropout_scope(self, condition: bool = True):
        """Context manager for enabling/disabling dropout during inference"""
        if condition:
            self.train()  # Enable dropout
        else:
            self.eval()  # Disable dropout
        try:
            yield
        finally:
            pass

    def predict(
        self,
        inputs: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        time: torch.Tensor = None,
        reshape_ensemble_dim: bool = True,
        num_predictions: int = 1,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict interpolated frames.

        Args:
            inputs: Input tensor [B, 2*C, H, W] (concatenated initial and target frames)
            condition: Static condition (unused in this simplified version)
            time: Time step for interpolation [B] or [B, 1]
            reshape_ensemble_dim: Whether to reshape ensemble dimension
            num_predictions: Number of predictions to generate

        Returns:
            Dictionary containing 'preds' key with interpolated frames
        """
        B = inputs.shape[0]

        # Ensure time is properly shaped
        if time.dim() == 1:
            time = time.unsqueeze(-1)  # [B, 1]

        # Encode the concatenated frames
        h = self.encoder(inputs)  # [B, hidden_dim, H, W]

        # Add time conditioning
        time_emb = self.time_proj(time.float())  # [B, hidden_dim]
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)  # [B, hidden_dim, 1, 1]
        h = h + time_emb

        # Apply dropout if in training mode
        if self.training:
            h = self.dropout(h)

        # Decode to get interpolated frame
        pred = self.decoder(h)  # [B, C, H, W]

        if num_predictions > 1:
            preds = []
            for _ in range(num_predictions):
                if self.training:
                    noise = torch.randn_like(pred) * 0.1
                    preds.append(pred + noise)
                else:
                    preds.append(pred)
            pred = torch.stack(preds, dim=1)  # [B, num_predictions, C, H, W]
            if reshape_ensemble_dim:
                pred = pred.reshape(B * num_predictions, *pred.shape[2:])

        return {"preds": pred}

    def forward(self, inputs: torch.Tensor, time: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for compatibility"""
        result = self.predict(inputs, time=time, **kwargs)
        return result["preds"]


def freeze_model(model: nn.Module):
    """Freeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def create_interpolator(channels: int, **kwargs) -> Interpolator:
    """Create a interpolator for DYffusion"""
    return Interpolator(channels=channels, **kwargs)
