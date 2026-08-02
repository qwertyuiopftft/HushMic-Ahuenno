from types import SimpleNamespace

from hushmic_personal_gpu import HybridSpeakerGate


def make_gate() -> HybridSpeakerGate:
    return HybridSpeakerGate(
        SimpleNamespace(
            ecapa_threshold=0.0,
            identity_fast_weight=1.6,
            identity_score_threshold=1.39,
            identity_reject_checks=2,
            verification_window_ms=2000,
        )
    )


def make_asymmetric_gate() -> HybridSpeakerGate:
    return HybridSpeakerGate(
        SimpleNamespace(
            ecapa_threshold=0.04,
            identity_fast_weight=2.25,
            identity_score_threshold=1.935,
            identity_close_fast_weight=3.0,
            identity_close_score_threshold=1.10,
            identity_reject_checks=2,
            verification_window_ms=2000,
        )
    )


def test_quiet_keeps_an_open_gate_open() -> None:
    gate = make_gate()
    gate.reset_after_quiet(preserve_closed=True)
    assert gate.is_open
    assert gate.ecapa_accepted is None


def test_quiet_does_not_undo_a_foreign_speaker_rejection() -> None:
    gate = make_gate()
    gate.update_ecapa(-1.0)
    gate.update_ecapa(-1.0)
    assert not gate.is_open
    gate.reset_after_quiet(preserve_closed=True)
    assert not gate.is_open
    assert gate.reject_count == 0
    gate.accept_short_window(0.7)
    assert gate.is_open


def test_open_gate_does_not_mute_an_uncertain_target() -> None:
    gate = make_asymmetric_gate()
    gate.last_fast_clip_probability = 0.40
    gate.update_ecapa(0.10)
    gate.update_ecapa(0.10)
    assert gate.last_identity_score < gate.identity_score_threshold
    assert gate.is_open
    assert gate.reject_count == 0


def test_open_gate_closes_on_confident_foreign_evidence() -> None:
    gate = make_asymmetric_gate()
    gate.last_fast_clip_probability = 0.20
    gate.update_ecapa(0.10)
    assert gate.is_open
    gate.update_ecapa(0.10)
    assert not gate.is_open


if __name__ == "__main__":
    test_quiet_keeps_an_open_gate_open()
    test_quiet_does_not_undo_a_foreign_speaker_rejection()
    test_open_gate_does_not_mute_an_uncertain_target()
    test_open_gate_closes_on_confident_foreign_evidence()
    print("HybridSpeakerGate quiet-latch tests: PASS")
