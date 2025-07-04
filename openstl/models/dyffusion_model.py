import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Sequence, Union, Dict
from tqdm import tqdm
from contextlib import ExitStack

from ..modules.interpolator import create_interpolator


class DYffusion_Model(nn.Module):
    r"""DYffusion Model for OpenSTL

    Adaptation of DYffusion from https://arxiv.org/abs/2306.01984, for use in OpenSTL framework
    """

    def __init__(
        self,
        in_shape,
        timesteps=None,
        schedule="before_t1_only",
        additional_interpolation_steps=0,
        additional_interpolation_steps_factor=0,
        forward_conditioning="data",
        sampling_type="cold",
        time_encoding="dynamics",
        interpolate_before_t1=True,
        enable_interpolator_dropout=True,
        use_cold_sampling_for_last_step=False,
        lambda_reconstruction=0.5,
        lambda_reconstruction2=0.5,
        refine_intermediate_predictions=False,
        prediction_timesteps=None,
        interpolator_hidden_dim=64,
        pre_seq_length=10,
        aft_seq_length=None,
        unet_version="unet_modules",
        **kwargs,
    ):
        super(DYffusion_Model, self).__init__()

        T_in, C, H, W = in_shape  # T_in is input sequence length
        self.channels = C
        self.image_size = H
        self.T_in = T_in  # Input sequence length
        self.T_out = aft_seq_length  # Output sequence length
        self.horizon = aft_seq_length
        self.self_condition = kwargs.get("self_condition", False)

        # Set timesteps to horizon if not specified (original DYffusion behavior)
        self.num_timesteps = timesteps if timesteps is not None else aft_seq_length
        self.original_timesteps = self.num_timesteps

        # Store hyperparameters
        self.schedule = schedule
        self.forward_conditioning = forward_conditioning
        self.sampling_type = sampling_type
        self.time_encoding = time_encoding
        self.interpolate_before_t1 = interpolate_before_t1
        self.enable_interpolator_dropout = enable_interpolator_dropout
        self.use_cold_sampling_for_last_step = use_cold_sampling_for_last_step
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_reconstruction2 = lambda_reconstruction2
        self.refine_intermediate_predictions = refine_intermediate_predictions
        self.prediction_timesteps = prediction_timesteps
        self.unet_version = unet_version

        # Build the backbone model (U-Net or similar)
        self.model = self._build_backbone(self.channels, **kwargs)

        # Build interpolator for intermediate steps
        self.interpolator = self._build_interpolator(C, interpolator_hidden_dim, **kwargs)

        # Initialize dynamic diffusion schedule
        self._setup_schedule(additional_interpolation_steps, additional_interpolation_steps_factor)

        # Loss function
        self.criterion = nn.MSELoss()

    def _build_backbone(self, channels, dim=64, **kwargs):
        """Build backbone model for dyffusion"""
        if self.unet_version == "resnet":
            from ..modules.unet_resnet import UNet
        elif self.unet_version == "simple":
            from ..modules.unet_simple import UNet
        else:
            raise ValueError(f"Unknown unet_version: {self.unet_version}")
        return UNet(
            dim=dim,
            channels=self.T_in * self.channels,  # Pass total channels for flattened sequence
            condition_channels=self.T_in
            * self.channels,  # Pass condition channels for flattened sequence
            self_condition=self.self_condition,
            **kwargs,
        )

    def _build_interpolator(self, channels, hidden_dim=64, **kwargs):
        """Build interpolator for intermediate frame prediction"""
        interpolator = create_interpolator(
            channels=channels, hidden_dim=hidden_dim, horizon=self.horizon, **kwargs
        )
        return interpolator

    def _setup_schedule(
        self, additional_interpolation_steps, additional_interpolation_steps_factor
    ):
        """Setup the dynamic diffusion schedule"""
        horizon = self.num_timesteps

        assert horizon > 1, f"horizon must be > 1, but got {horizon}"

        if self.schedule == "linear":
            assert additional_interpolation_steps == 0, (
                "additional_interpolation_steps must be 0 when using linear schedule"
            )
            self.additional_interpolation_steps_fac = additional_interpolation_steps_factor
            if self.interpolate_before_t1:
                interpolated_steps = horizon - 1
                self.di_to_ti_add = 0
            else:
                interpolated_steps = horizon - 2
                self.di_to_ti_add = additional_interpolation_steps_factor
            self.additional_diffusion_steps = (
                additional_interpolation_steps_factor * interpolated_steps
            )
        elif self.schedule == "before_t1_only":
            assert additional_interpolation_steps_factor == 0, (
                "additional_interpolation_steps_factor must be 0 when using before_t1_only schedule"
            )
            assert self.interpolate_before_t1, (
                "interpolate_before_t1 must be True when using before_t1_only schedule"
            )
            self.additional_diffusion_steps = additional_interpolation_steps
        else:
            raise ValueError(f"Invalid schedule: {self.schedule}")

        self.num_timesteps += self.additional_diffusion_steps

        # Create mapping between diffusion and interpolation steps
        d_to_i_step = {
            d: self.diffusion_step_to_interpolation_step(d) for d in range(1, self.num_timesteps)
        }
        self.dynamical_steps = {d: i_n for d, i_n in d_to_i_step.items() if float(i_n).is_integer()}
        self.artificial_interpolation_steps = {
            d: i_n for d, i_n in d_to_i_step.items() if not float(i_n).is_integer()
        }

    def diffusion_step_to_interpolation_step(
        self, diffusion_step: Union[int, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        """
        Convert a diffusion step to an interpolation step.
        This is the core mapping function from the original DYffusion.
        """
        # Validate range
        if torch.is_tensor(diffusion_step):
            assert (0 <= diffusion_step).all() and (diffusion_step <= self.num_timesteps - 1).all()
        else:
            assert 0 <= diffusion_step <= self.num_timesteps - 1

        if self.schedule == "linear":
            i_n = (diffusion_step + self.di_to_ti_add) / (
                self.additional_interpolation_steps_fac + 1
            )
        elif self.schedule == "before_t1_only":
            if torch.is_tensor(diffusion_step):
                i_n = torch.where(
                    diffusion_step >= self.additional_diffusion_steps + 1,
                    (diffusion_step - self.additional_diffusion_steps).float(),
                    diffusion_step / (self.additional_diffusion_steps + 1),
                )
            elif diffusion_step >= self.additional_diffusion_steps + 1:
                i_n = diffusion_step - self.additional_diffusion_steps
            else:
                i_n = diffusion_step / (self.additional_diffusion_steps + 1)
        else:
            raise ValueError(f"schedule='{self.schedule}' not supported.")

        return i_n

    def q_sample(
        self,
        x0,
        x_end,
        t: Optional[torch.Tensor] = None,
        interpolation_time: Optional[torch.Tensor] = None,
        is_artificial_step: bool = True,
        static_condition: Optional[torch.Tensor] = None,
        num_predictions: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample from q(x_t | x_0, x_end) using the interpolator.
        This is the core interpolation function.
        """
        assert t is None or interpolation_time is None, (
            "Either t or interpolation_time must be None."
        )
        t = interpolation_time if t is None else self.diffusion_step_to_interpolation_step(t)

        # Enable dropout during training or if explicitly enabled
        do_enable = self.training or self.enable_interpolator_dropout

        with ExitStack() as stack:
            stack.enter_context(self.interpolator.inference_dropout_scope(condition=do_enable))
            x_ti = self._interpolate(
                initial_condition=x_end,
                x_last=x0,
                t=t,
                static_condition=static_condition,
                num_predictions=num_predictions,
                **kwargs,
            )
        return x_ti

    def _interpolate(
        self,
        initial_condition: torch.Tensor,
        x_last: torch.Tensor,
        t: torch.Tensor,
        static_condition: Optional[torch.Tensor] = None,
        num_predictions: int = 1,
        **kwargs,
    ):
        """Internal interpolation method using the interpolator network"""
        # Ensure time is in valid range
        assert (0 < t).all() and (t < self.horizon).all(), (
            f"interpolate time must be in (0, {self.horizon}), got {t}"
        )

        # Prepare inputs for interpolator: concatenate individual frames
        B = initial_condition.shape[0]

        # Handle flattened sequence data: extract individual frames
        if initial_condition.dim() == 4:  # [B, T*C, H, W] - flattened sequence
            if initial_condition.shape[1] == self.channels:
                # Already single frame
                initial_frame = initial_condition
            else:
                # Extract last frame from flattened sequence
                initial_frame = initial_condition[:, -self.channels :]  # Take last frame
        elif initial_condition.dim() == 5:  # [B, T, C, H, W]
            initial_frame = initial_condition[:, -1]  # Take last frame
        else:
            initial_frame = initial_condition

        if x_last.dim() == 4:  # [B, T*C, H, W] - flattened sequence
            if x_last.shape[1] == self.channels:
                # Already single frame
                target_frame = x_last
            else:
                # Extract last frame from flattened sequence
                target_frame = x_last[:, -self.channels :]  # Take last frame
        elif x_last.dim() == 5:
            target_frame = x_last[:, -1]  # Take last frame
        else:
            target_frame = x_last

        interpolator_inputs = torch.cat([initial_frame, target_frame], dim=1)

        interpolator_outputs = self.interpolator.predict(
            interpolator_inputs,
            condition=static_condition,
            time=t,
            num_predictions=num_predictions,
            **kwargs,
        )

        interpolated_frame = interpolator_outputs["preds"]  # [B, C, H, W]

        if initial_condition.dim() == 4 and initial_condition.shape[1] > self.channels:
            seq_length = initial_condition.shape[1] // self.channels
            interpolated_sequence = interpolated_frame.repeat(1, seq_length, 1, 1)
            return interpolated_sequence
        else:
            return interpolated_frame

    def get_time_encoding(self, timestep, interpolation_step):
        """Get time encoding for the model"""
        if self.time_encoding == "dynamics":
            return interpolation_step
        elif self.time_encoding == "diffusion":
            return timestep
        elif self.time_encoding == "discrete":
            return timestep
        elif self.time_encoding == "normalized":
            return timestep / self.num_timesteps
        else:
            raise ValueError(f"Invalid time_encoding: {self.time_encoding}")

    def _predict_last_dynamics(
        self, forward_condition: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ):
        """Predict the final dynamics using the backbone model"""
        if self.time_encoding == "discrete":
            time = t
        elif self.time_encoding == "normalized":
            time = t / self.num_timesteps
        elif self.time_encoding == "dynamics":
            time = self.diffusion_step_to_interpolation_step(t)
        else:
            raise ValueError(f"Invalid time_encoding: {self.time_encoding}")

        x_last_pred = self.model(x_t, time=time, condition=forward_condition)
        return x_last_pred

    def predict_x_last(
        self,
        condition: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        is_sampling: bool = False,
        static_condition: Optional[torch.Tensor] = None,
    ):
        """Predict x_last using the forecasting model (backbone)"""
        assert (0 <= t).all() and (t <= self.num_timesteps - 1).all(), f"Invalid timestep: {t}"

        cond_type = self.forward_conditioning
        if cond_type == "data":
            forward_cond = condition
        elif cond_type == "none":
            forward_cond = None
        elif "data+noise" in cond_type:
            # Linear combination of condition and noise based on timestep
            tfactor = t / (self.num_timesteps - 1)
            tfactor = tfactor.view(condition.shape[0], *[1] * (condition.ndim - 1))
            forward_cond = tfactor * condition + (1 - tfactor) * torch.randn_like(condition)
        else:
            raise ValueError(f"Invalid forward conditioning type: {cond_type}")

        # Get condition (combine with static condition if available)
        forward_cond = self.get_condition(
            condition=forward_cond,
            x_last=None,
            prediction_type="forward",
            static_condition=static_condition,
            shape=condition.shape if condition is not None else None,
        )

        x_last_pred = self._predict_last_dynamics(x_t=x_t, forward_condition=forward_cond, t=t)
        return x_last_pred

    def get_condition(
        self,
        condition,
        x_last: Optional[torch.Tensor],
        prediction_type: str,
        static_condition: Optional[torch.Tensor] = None,
        shape: Sequence[int] = None,
    ) -> torch.Tensor:
        """Combine different types of conditions"""
        if static_condition is None:
            return condition
        elif condition is None:
            return static_condition
        else:
            return torch.cat([condition, static_condition], dim=1)

    def p_losses(
        self,
        xt_last: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        static_condition: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the DYffusion training losses.
        This implements the dual-loss training from the original paper.
        """
        lam1 = self.lambda_reconstruction
        lam2 = self.lambda_reconstruction2

        # Create the inputs for the forecasting model
        # 1. For t=0, simply use the initial conditions
        x_t = torch.zeros_like(xt_last)

        # 2. For t>0, we need to interpolate the data using the interpolator
        t_nonzero = t > 0
        if t_nonzero.any():
            x_interpolated = self.q_sample(
                x_end=condition[t_nonzero],
                x0=xt_last[t_nonzero],
                t=t[t_nonzero],
                static_condition=None if static_condition is None else static_condition[t_nonzero],
                num_predictions=1,
            )
            x_t[t_nonzero] = x_interpolated.to(x_t.dtype)

        t_zero = t == 0
        if t_zero.any():
            x_t[t_zero] = condition[t_zero].to(x_t.dtype)

        # Train the forward predictions (predict xt_last from x_t)
        xt_last_target = xt_last
        xt_last_pred = self.predict_x_last(
            condition=condition, x_t=x_t, t=t, static_condition=static_condition
        )
        loss_forward = self.criterion(xt_last_pred, xt_last_target)

        # Train the forward predictions II by emulating one more step
        tnot_last = t <= self.num_timesteps - 2
        t2 = t[tnot_last] + 1
        calc_t2 = tnot_last.any()

        if lam2 > 0 and calc_t2:
            cond_notlast = condition[tnot_last]
            x0not_last = xt_last_pred[tnot_last]
            sc_notlast = None if static_condition is None else static_condition[tnot_last]

            # Use predictions to interpolate the next step
            x_interpolated2 = self.q_sample(
                x_end=cond_notlast,
                x0=x0not_last,
                t=t2,
                static_condition=sc_notlast,
                num_predictions=1,
            )
            x_last_pred2 = self.predict_x_last(
                condition=cond_notlast, x_t=x_interpolated2, t=t2, static_condition=sc_notlast
            )
            loss_forward2 = self.criterion(x_last_pred2, xt_last_target[tnot_last])
        else:
            loss_forward2 = torch.tensor(0.0, device=xt_last.device)

        loss = lam1 * loss_forward + lam2 * loss_forward2

        return {
            "loss": loss,
            "loss_forward": loss_forward,
            "loss_forward2": loss_forward2,
        }

    def sample_loop(
        self,
        initial_condition,
        static_condition: Optional[torch.Tensor] = None,
        num_predictions: int = 1,
    ):
        """
        Main sampling loop implementing the DYffusion algorithm.
        This is the core inference method from the original implementation.
        """
        batch_size = initial_condition.shape[0]
        device = initial_condition.device

        sc_kw = dict(static_condition=static_condition)
        assert len(initial_condition.shape) == 4, (
            f"condition.shape: {initial_condition.shape} (should be 4D)"
        )

        # Initialize the target sequence with noise
        # initial_condition is [B, T_in*C, H, W], we need [B, T_out*C, H, W] for prediction
        B, T_in_C, H, W = initial_condition.shape
        T_out_C = T_in_C  # Assume same number of output channels as input

        # Start from noise for the target sequence
        x_s = torch.randn(B, T_out_C, H, W, device=device, dtype=initial_condition.dtype)
        intermediates, x0_hat, dynamics_pred_step = dict(), None, 0

        # Create sampling schedule (simplified - use all timesteps)
        sampling_schedule = list(range(0, self.num_timesteps))
        last_i_n_plus_one = sampling_schedule[-1] + 1

        s_and_snext = zip(
            sampling_schedule,
            sampling_schedule[1:] + [last_i_n_plus_one],
            sampling_schedule[2:] + [last_i_n_plus_one, last_i_n_plus_one],
        )

        for s, s_next, s_nnext in tqdm(s_and_snext, desc="DYffusion Sampling", leave=False):
            is_last_step = s == self.num_timesteps - 1

            # F(x_s, s) = predict target data
            step_s = torch.full((batch_size,), s, dtype=torch.float32, device=device)
            x0_hat = self.predict_x_last(
                condition=initial_condition, x_t=x_s, t=step_s, is_sampling=True, **sc_kw
            )

            # Are we predicting dynamical time step or artificial interpolation step?
            time_i_n = (
                self.diffusion_step_to_interpolation_step(s_next) if not is_last_step else np.inf
            )
            is_dynamics_pred = float(time_i_n).is_integer() or is_last_step

            q_sample_kwargs = dict(
                x0=x0_hat,
                x_end=initial_condition,
                is_artificial_step=not is_dynamics_pred,
                num_predictions=1 if is_last_step else num_predictions,
            )

            if s_next <= self.num_timesteps - 1:
                # D(x_s, s-1)
                step_s_next = torch.full((batch_size,), s_next, dtype=torch.float32, device=device)
                x_interpolated_s_next = self.q_sample(**q_sample_kwargs, t=step_s_next, **sc_kw)
            else:
                x_interpolated_s_next = x0_hat

            # Apply sampling strategy
            if self.sampling_type == "cold":
                if is_last_step and not self.use_cold_sampling_for_last_step:
                    x_s = x0_hat
                else:
                    # Cold sampling: x_s = x_s - D(x_s, s) + D(x_s, s-1)
                    x_interpolated_s = (
                        self.q_sample(**q_sample_kwargs, t=step_s, **sc_kw) if s > 0 else x_s
                    )
                    x_s = x_s - x_interpolated_s + x_interpolated_s_next
            elif self.sampling_type == "naive":
                x_s = x_interpolated_s_next
            else:
                raise ValueError(f"Unknown sampling type {self.sampling_type}")

            dynamics_pred_step = (
                int(time_i_n) if s < self.num_timesteps - 1 else dynamics_pred_step + 1
            )
            if is_dynamics_pred:
                intermediates[f"t{dynamics_pred_step}_preds"] = x_s

        if last_i_n_plus_one < self.num_timesteps:
            return x_s, intermediates, x_interpolated_s_next
        return x0_hat, intermediates, x_s

    def forward_training(self, batch_x, batch_y):
        """Forward pass during training using DYffusion p_losses"""
        B, T_in, C, H, W = batch_x.shape
        B, T_out, C, H, W = batch_y.shape
        device = batch_x.device

        # Random timestep for diffusion
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()

        # Use DYffusion training loss
        loss_dict = self.p_losses(
            xt_last=batch_y.view(B, T_out * C, H, W),  # Flatten sequence dimension
            condition=batch_x.view(B, T_in * C, H, W),  # Flatten sequence dimension
            t=t,
        )

        return loss_dict

    def forward_inference(self, batch_x, aft_seq_length):
        """Forward pass during inference using DYffusion sampling"""
        B, T_in, C, H, W = batch_x.shape

        # Use entire input sequence as initial condition (flattened like in training)
        initial_condition = batch_x.view(B, T_in * C, H, W)  # [B, T_in*C, H, W]

        # Sample using DYffusion algorithm to predict next sequence
        x_pred, intermediates, _ = self.sample_loop(initial_condition)

        # Reshape prediction back to sequence format
        pred_y = x_pred.view(B, aft_seq_length, C, H, W)  # [B, T_out, C, H, W]
        return pred_y

    def forward(self, x, **kwargs):
        """Placeholder forward pass for compatibility"""
        return x
