"""Configure the CUDA allocator before importing the training dependencies."""

import os


def configure_allocator():
    if not any(key in os.environ for key in
               ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def train_main(argv=None):
    configure_allocator()
    from svct_pl.lightning.official_cli import main
    return main(argv)


def benchmark_main(argv=None):
    configure_allocator()
    from svct_pl.lightning.memory_benchmark import main
    return main(argv)
