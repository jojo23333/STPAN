from yacs.config import CfgNode as CN

_C = CN()

# Model definition
_C.MODEL = CN()


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

_C.TRAIN.ENABLE = False

# Train dataset
_C.TRAIN.DATASET = 'nitre'

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 16

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 3

_C.TRAIN.USE_CENTER_VALIDATION = False

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""


_C.TRAIN.LOAD_PART_OF_CHECKPOINT = False

# Spatial patch size when training
_C.TRAIN.SP_PATCH_SIZE = 128


# ---------------------------------------------------------------------------- #
# Validation options.
# ---------------------------------------------------------------------------- #
_C.VAL = CN()

_C.VAL.VAL_PATCH_SIZE = 512


# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# Test dataset
_C.TEST.DATASET = 'nitre'

_C.TEST.BATCH_SIZE = 1

_C.TEST.CHECKPOINT_FILE_PATH = ""

_C.TEST.OUTPUT_DIR = "./test_out"

_C.TEST.SAVE_IMG = True

_C.TEST.TEST_PATCH_SIZE = 512

_C.TEST.AUGMENT_TEST = False

_C.TEST.TEST_MODE = 'part'

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# model architecture
_C.MODEL.ARCH = "dncnn"

_C.MODEL.LOSS_FUNC = "l2"


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.OPTIMIZING_METHOD = "adam"

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 0

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Base learning rate.
_C.SOLVER.BASE_LR = 4e-4

_C.SOLVER.MIN_LR = 1e-5

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "steps_with_relative_lrs"

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = [0, 30, 50, 60]

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = [1, 0.1, 0.01, 0.001]

_C.SOLVER.MAX_EPOCH = 300

_C.SOLVER.USE_DCT_MASK = True

_C.SOLVER.DCT_WARM_UP = 5

_C.SOLVER.DCT_MASK_STEPS = [10, 30]


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CN()

_C.DATA.NUM_FRAMES = 5

# The path to the data directory.
_C.DATA.PATH_TO_TRAINING_SET = "/Disk2/limuchen/NTIRE2020/train"

_C.DATA.PATH_TO_TEST_SET = "/Disk2/limuchen/NTIRE2020/val"

#
_C.DATA.DATA_SHAPE = (3, 1920, 1080)

# -----------------------------------------------------------------------------
# Dataloader options
# -----------------------------------------------------------------------------
_C.DATA_LOADER = CN()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------------------------------------------------- #
# UNET options
# ---------------------------------------------------------------------------- #

_C.UNET = CN()

_C.UNET.NUM_FILTERS = [64, 64, 128, 256]

_C.UNET.USE_DENSE_BLOCK = False

_C.UNET.USE_GLOBAL_FUSION = True

_C.UNET.NUM_FUSION_NL_GROUPS = 4


# ---------------------------------------------------------------------------- #
# EDVR options
# ---------------------------------------------------------------------------- #

_C.EDVR = CN()

_C.EDVR.NF = 64

_C.EDVR.BACK_RBS = 10

_C.EDVR.PREDEBLUR = False

_C.EDVR.USE_GLOBAL_FUSION = True

_C.EDVR.USE_CARAFE = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# Distributed backend.
_C.DIST_BACKEND = "nccl"

_C.LOG_NAME = 'log'

_C.TASK = 'denoise'


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()