import random
import numpy as np
import torch


def set_global_seed(seed: int = 6304):
    """Seed python, numpy and torch RNGs. Call once at the start of any
    training/eval script. Deterministic per-image randomness (e.g. patch
    shuffle) uses its own seeded RandomState instances instead of relying
    on this global state, so those results don't depend on call order.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if(torch.mps.is_available()):
        torch.mps.manual_seed(seed)
    if(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)
