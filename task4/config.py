"""
Minimal YAML config loading for Task 4, matching the assignment's
recommended repo structure (task4/configs/vanilla.yaml, gcsc.yaml,
proser.yaml). Each config carries the hyperparameters that section of the
spec fixes (optimizer, epochs, batch size, method-specific loss weights),
so a run's exact settings live in a versioned file rather than being
buried in code -- per the assignment's instruction to "preserve the
configurations used to produce your results."
"""

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}
