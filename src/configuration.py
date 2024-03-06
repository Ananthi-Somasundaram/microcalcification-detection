import argparse
import torch

from dataclasses import dataclass


def parse_environment_args(argument_parser: argparse.ArgumentParser) -> dict:
    args = argument_parser.parse_args()
    
    keys: list[str] = args.env[::2]  # Every even element is a key
    vals: list[str] = args.env[1::2]  # Every odd element is a value

    configuration_arguments: dict = {k: int(v) if v.isdigit() else v for k, v in zip(keys, vals)}

    return configuration_arguments


@dataclass
class Configuration():
    batch_size: int = 40
    test_batch_size: int = 400
    validate_batch_size: int = 1000
    epochs: int = 350
    epochs_validate: int = 1000
    learning_rate: float = 0.001
    momentum: float = 0.8
    no_cuda: bool = False
    seed: int = 1
    log_interval: int = 40

    def __post_init__(self):
        self.cuda: bool = not self.no_cuda and torch.cuda.is_available() # Returns a bool indicating if CUDA is currently available

