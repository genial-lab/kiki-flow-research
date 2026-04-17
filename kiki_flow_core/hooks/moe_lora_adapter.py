"""Read-only adapter to micro-kiki MoE-LoRA stack states."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class MoELoraAdapter:
    """Read snapshot of LoRA stack states. Read-only in v1."""

    def __init__(self, snapshotter: Callable[[], dict[str, np.ndarray]]) -> None:
        self.snapshotter = snapshotter

    def snapshot_stack_states(self) -> dict[str, np.ndarray]:
        return dict(self.snapshotter())

    def stack_names(self) -> list[str]:
        return list(self.snapshotter().keys())
