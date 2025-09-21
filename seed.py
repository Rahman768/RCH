import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed: int = 42, cudnn_deterministic: bool = True, cudnn_benchmark: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = cudnn_deterministic
    cudnn.benchmark = cudnn_benchmark
