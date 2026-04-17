"""Drift heuristic: fires when new-concept rate in recent Aeon exceeds threshold."""

from __future__ import annotations

from typing import Any

from kiki_flow_core.hooks import AeonAdapter


class DriftTrigger:
    """Fires when |new_concepts| / |total_concepts_in_window| > threshold."""

    def __init__(self, threshold: float = 0.05, window_h: int = 24) -> None:
        self.threshold = threshold
        self.window_h = window_h

    def should_fire(self, aeon: AeonAdapter, last_snapshot_manifest: dict[str, Any]) -> bool:
        episodes = aeon.fetch_recent_episodes(window_h=self.window_h)
        if not episodes:
            return False
        all_concepts: set[str] = set()
        for ep in episodes:
            all_concepts.update(ep.get("concepts", []))
        if not all_concepts:
            return False
        known = set(last_snapshot_manifest.get("known_concepts", []))
        new = all_concepts - known
        return len(new) / len(all_concepts) > self.threshold
