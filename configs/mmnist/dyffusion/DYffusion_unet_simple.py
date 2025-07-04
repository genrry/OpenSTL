method = 'dyffusion'
model_type = 'simple'
# DYffusion specific parameters (matching original defaults)
aft_seq_length = 10  # Original default is 10
timesteps = None  # Auto-inferred from datamodule.horizon
schedule = 'before_t1_only'
additional_interpolation_steps = 0  # Original default is 0
additional_interpolation_steps_factor = 0
forward_conditioning = 'data'
sampling_type = 'cold'
time_encoding = 'dynamics'
interpolate_before_t1 = True
enable_interpolator_dropout = True
use_cold_sampling_for_last_step = False

# Loss weights (core DYffusion parameters)
lambda_reconstruction = 0.5    # Main loss term
lambda_reconstruction2 = 0.5   # Auxiliary loss term

# Sampling and refinement
refine_intermediate_predictions = False
prediction_timesteps = None
sampling_schedule = None  # Use all timesteps

# UNet model parameters  
dim = 64
dim_mults = [2, 2, 4, 4, 8, 8]  
has_time_emb = True
unet_version = 'simple' # 'simple' or 'resnet'

# Interpolator parameters
interpolator_hidden_dim = 64

# Training parameters
lr = 1e-4
batch_size = 16 
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 5

# Validation setting
use_inference_for_validation = False  # Use training losses for validation

# Optimization specific to DYffusion
weight_decay = 0.01
grad_clip_val = 1.0
