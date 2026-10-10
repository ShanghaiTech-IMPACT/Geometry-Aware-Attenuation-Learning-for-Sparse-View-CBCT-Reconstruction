"""Lightning training entry point; train.py retains the upstream trainer."""

import os


def configure_allocator():
    if not any(key in os.environ for key in
               ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main(argv=None):
    configure_allocator()
    from svct_pl.lightning.official_cli import main as run_training
    return run_training(argv)


if __name__ == "__main__":
    main()
