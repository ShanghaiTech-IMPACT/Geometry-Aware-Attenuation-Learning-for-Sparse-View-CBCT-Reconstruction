from contextlib import contextmanager, nullcontext

from torch import nn


@contextmanager
def _restore_bn_buffers_after_recompute(module):
    """Let recompute follow the identical BN graph, then undo buffer updates.

    Changing ``track_running_stats`` would change the recompute operator graph and
    can fail non-reentrant checkpoint determinism checks. Snapshot/restore keeps
    the graph identical while making the second update externally invisible.
    """
    batch_norms = [item for item in module.modules()
                   if isinstance(item, nn.modules.batchnorm._BatchNorm)]
    states = []
    for item in batch_norms:
        # Keep the once-updated forward buffers and let recompute mutate temporary
        # clones. Reassignment (instead of copy_) also avoids version-counter
        # changes to tensors that a backend may retain for backward.
        states.append((item.running_mean, item.running_var, item.num_batches_tracked))
        if item.running_mean is not None:
            item.running_mean = item.running_mean.clone()
        if item.running_var is not None:
            item.running_var = item.running_var.clone()
        if item.num_batches_tracked is not None:
            item.num_batches_tracked = item.num_batches_tracked.clone()
    try:
        yield
    finally:
        for item, (mean, var, count) in zip(batch_norms, states):
            item.running_mean, item.running_var = mean, var
            item.num_batches_tracked = count


def checkpoint_context_fn(module):
    """Context pair for torch checkpoint: normal forward, BN-safe recompute."""
    return nullcontext(), _restore_bn_buffers_after_recompute(module)
