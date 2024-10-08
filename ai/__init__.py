import torch

import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"