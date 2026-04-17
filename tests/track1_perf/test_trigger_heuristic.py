from kiki_flow_core.hooks import AeonAdapter
from kiki_flow_core.track1_perf.trigger_heuristic import DriftTrigger


def test_drift_trigger_fires_when_new_concepts_above_threshold():
    aeon = AeonAdapter(fetcher=lambda h: [{"concepts": ["a", "b", "c", "d", "e"]}])
    manifest = {"known_concepts": ["a", "b"]}  # 3 new of 5 => 60% drift
    trigger = DriftTrigger(threshold=0.2)
    assert trigger.should_fire(aeon, manifest) is True


def test_drift_trigger_holds_when_low_drift():
    aeon = AeonAdapter(fetcher=lambda h: [{"concepts": ["a", "b", "c"]}])
    manifest = {"known_concepts": ["a", "b", "c"]}  # 0% drift
    trigger = DriftTrigger(threshold=0.2)
    assert trigger.should_fire(aeon, manifest) is False


def test_drift_trigger_handles_empty_aeon():
    aeon = AeonAdapter(fetcher=lambda h: [])
    trigger = DriftTrigger(threshold=0.05)
    assert trigger.should_fire(aeon, {"known_concepts": []}) is False
