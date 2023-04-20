import os
from pathlib import Path

from colossalai.amp import AMP_TYPE

# ==== Colossal-AI Configuration ====

clip_grad_norm = 2.0
gradient_accumulation = 5
fp16 = dict(mode=AMP_TYPE.TORCH)

DATAPATH = Path(__file__).parent.parent / "ecg_data"
EEGPATH = Path(__file__).parent.parent / "chb-mit-scalp-eeg"
# DATAPATH = r'/home/wangruopeng/pretrain-data/icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0'
# ==== Model Configuration ====
#
# Variable Naming Convension:
#
# 1. `THIS_WILL_BE_DERECTLY_ACCESSED_BY_MAIN`: All capital.
#   eg: VERBOSE, LEARNING_RATE
#
# 2. `_THIS_WILL_BE_USED_TO_GENERATE_(1)`: Begin with underscore.
#   eg: __BASE_LEARNING_RATE
#
# 3. `this_is_a_simple_helper`: Snake case.
#   eg: eff_batch_size

# toggle more loggings
VERBOSE = False
DEBUG = False

NUM_EPOCHS = 80
# epochs to warmup LR
WARMUP_EPOCHS = 40 if NUM_EPOCHS > 1 else 0

# Interval to save a checkpoint
CHECKPOINT_INTERVAL = 1

# Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus
BATCH_SIZE = 16

#  SummaryWriter dir
SummaryWriter_DIR = Path(__file__).parent / "runs/base_convheader"

# Place to save pretrained model
OUTPUT_DIR = Path(__file__).parent.parent / "output_mae_base_convheader"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Masking ratio (percentage of removed patches).
MASK_RATIO = 0.75

# learning rate (absolute lr)
LEARNING_RATE = 1.5e-4
# lower lr bound for cyclic schedulers that hit 0
MINIMUM_LEARNING_RATE = 0.0001
# base learning rate: absolute_lr = base_lr * total_batch_size / 256
_BASE_LEARNING_RATE = 0.002

WEIGHT_DECAY = 0.05

# Use (per-patch) normalized pixels as targets for computing loss
NORM_PIX_LOSS = False
# torch_ddp = dict(find_unused_parameters=True)  # morlet and resample need this
# resume from checkpoint
RESUME = False
if RESUME:
    RESUME_DIR = Path(__file__).parent / "output_mae/checkpoint-41.pth"
