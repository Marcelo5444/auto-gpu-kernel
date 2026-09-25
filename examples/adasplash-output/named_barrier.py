"""Named-barrier IDs for the AdaSplash forward kernels.

Mirrors ``flash_attn/cute/named_barrier.py``: one IntEnum per kernel, values
explicit rather than ``enum.auto()`` because these kernels index *ranges* off a
base (``MaskMergeBase + warp_group_idx``), so the numbering is load-bearing.

Hardware has 16 barriers.  0 is reserved for ``sync_threads()`` and 1-6 belong to
``NamedBarrierFwd`` upstream -- of which only the scheduler pair 2/3 can be live
in these kernels -- so everything below starts at 7.
"""

import enum


class NamedBarrierTauHist(enum.IntEnum):
    """Kernel A (``fwd_tau_hist``) uses only the inherited scheduler pair."""


class NamedBarrierTauRefine(enum.IntEnum):
    """Kernel B (``fwd_tau_refine``): tile-end mask merge, one slot per warpgroup."""

    ## 7/8 = "replica stores visible"
    MaskMergeBase = 7
    ## 9/10 = "replica reads + sidecar partials done"; also gates the next
    ## tile's re-zero of the replicas.
    MaskDoneBase = 9


class NamedBarrierOutput(enum.IntEnum):
    """Kernel C (``fwd_output``)."""

    ProducerDecode = 9  # producer-internal decode scan (128 threads)