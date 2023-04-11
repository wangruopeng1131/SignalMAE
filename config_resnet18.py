import os
from pathlib import Path

from colossalai.amp import AMP_TYPE

# ==== Colossal-AI Configuration ====

clip_grad_norm = 1.0
gradient_accumulation = 1
fp16 = dict(mode=AMP_TYPE.TORCH)

DATA_PATH = r'/home/wangruopeng/database/noise_data'

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
VERBOSE = True
DEBUG = False

NUM_EPOCHS = 2
# epochs to warmup LR
WARMUP_EPOCHS = 1 if NUM_EPOCHS > 1 else 0

# Interval to save a checkpoint
CHECKPOINT_INTERVAL = 1

# Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus
BATCH_SIZE = 128

# Place to save pretrained model
OUTPUT_DIR = Path(__file__).parent.parent / "output_resnet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# learning rate (absolute lr)
LEARNING_RATE = 1e-6
# lower lr bound for cyclic schedulers that hit 0
MINIMUM_LEARNING_RATE = 0
# base learning rate: absolute_lr = base_lr * total_batch_size / 256
_BASE_LEARNING_RATE = 1e-3

WEIGHT_DECAY = 0.5

# Use (per-patch) normalized pixels as targets for computing loss
NORM_PIX_LOSS = True

# resume from checkpoint
RESUME = False
if RESUME:
    RESUME_DIR = "/home/wangruopeng/ppg_bp/output_resnet/checkpoint-5.pth"