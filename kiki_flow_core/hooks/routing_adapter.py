"""Advisory-only routing publisher. Never raises."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("kiki_flow.routing")


class RoutingAdapter:
    """Publish advisory routing suggestions to micro-kiki via a callback; swallow exceptions."""

    def __init__(self, publisher: Callable[[dict[str, Any]], None]) -> None:
        self.publisher = publisher

    def publish_advisory(self, advisory: dict[str, Any]) -> None:
        try:
            self.publisher(advisory)
        except Exception as e:  # noqa: BLE001 - advisory must never crash caller
            logger.warning("routing advisory swallowed: %s", e)
