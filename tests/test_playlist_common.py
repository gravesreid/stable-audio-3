"""Tests for the normalization in playlist_common.py.

The mellow playlist's whole premise is that you can leave it on, which fails if
track 45 is 20 dB quieter than track 44. That levelling happens in normalize_wav,
so these check it on synthetic signals -- a dense pad and a sparse piano attack
are just a sine and a decaying click. No model, no weights.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playlist_common import (  # noqa: E402
    CEILING,
    RMS_MAX_BOOST_DB,
    RMS_TARGET_DBFS,
    SOFT_CLIP_KNEE,
    normalize_wav,
    soft_clip,
)

SR = 44100


def _dbfs(wav) -> float:
    return 20 * math.log10(wav.pow(2).mean().sqrt().item() + 1e-12)


def _dense(seconds: float = 2.0, amplitude: float = 0.5):
    """A sustained tone: a small crest factor, like a pad or a full mix."""
    t = torch.arange(int(SR * seconds)) / SR
    return (amplitude * torch.sin(2 * math.pi * 220 * t)).unsqueeze(0)


def _sparse(seconds: float = 2.0, hits: int = 4):
    """Loud short attacks over near-silence: a big crest factor, like felt piano."""
    n = int(SR * seconds)
    t = torch.arange(n) / SR
    wav = torch.zeros(n)
    for i in range(hits):
        start = int(i * n / hits)
        env = torch.exp(-torch.arange(n - start) / (0.02 * SR))
        wav[start:] += 0.9 * env * torch.sin(2 * math.pi * 440 * t[: n - start])
    return wav.unsqueeze(0)


# ---------------------------------------------------------------------------
# soft_clip
# ---------------------------------------------------------------------------


def test_soft_clip_leaves_quiet_signal_untouched():
    wav = _dense(amplitude=SOFT_CLIP_KNEE * 0.9)
    assert torch.allclose(soft_clip(wav), wav)


# float32 tanh saturates to exactly 1.0, so the ceiling is reached rather than
# merely approached. Anything at or under it is safe; over it is a clipped file.
TOL = 1e-6


def test_soft_clip_keeps_everything_under_the_ceiling():
    wav = _dense(amplitude=8.0)  # wildly over full scale
    assert soft_clip(wav).abs().max().item() <= CEILING + TOL


def test_soft_clip_is_monotonic_and_odd():
    x = torch.linspace(-4, 4, 2001).unsqueeze(0)
    y = soft_clip(x)
    assert torch.all(y.diff() >= 0), "soft clip must not fold the waveform back"
    assert torch.allclose(y, -soft_clip(-x), atol=TOL)


def test_soft_clip_still_resolves_detail_below_the_ceiling():
    """Saturation is fine at the tips; it must not flatten the useful range."""
    x = torch.linspace(0, 1.2, 1201).unsqueeze(0)
    assert torch.all(soft_clip(x).diff() > 0)


def test_soft_clip_has_no_step_at_the_knee():
    """A discontinuity at the knee would be audible as distortion."""
    x = torch.tensor([[SOFT_CLIP_KNEE - 1e-4, SOFT_CLIP_KNEE + 1e-4]])
    y = soft_clip(x)
    assert y.diff().abs().item() < 1e-3


# ---------------------------------------------------------------------------
# normalize_wav
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["peak", "rms"])
def test_silence_survives_intact(mode):
    """A dead generation must not come back as NaN from a divide by zero."""
    wav = torch.zeros(1, SR)
    out = normalize_wav(wav, mode)
    assert torch.equal(out, wav)


def test_peak_mode_hits_the_ceiling():
    for wav in (_dense(amplitude=0.02), _sparse()):
        assert normalize_wav(wav, "peak").abs().max().item() == pytest.approx(CEILING)


@pytest.mark.parametrize("mode", ["peak", "rms"])
def test_nothing_ever_exceeds_the_ceiling(mode):
    """Above 1.0 the wav file clips, which is the least mellow sound there is."""
    for wav in (_dense(amplitude=0.001), _dense(amplitude=0.9), _sparse()):
        assert normalize_wav(wav, mode).abs().max().item() <= CEILING + TOL


def test_rms_mode_hits_the_loudness_target_when_it_has_the_headroom():
    out = normalize_wav(_dense(amplitude=0.02), "rms")
    assert _dbfs(out) == pytest.approx(RMS_TARGET_DBFS, abs=0.5)


def test_rms_mode_levels_sparse_against_dense_far_better_than_peak():
    """The bug this exists for: peak-matched sparse and dense material is not
    level-matched, and the gap is enormous."""
    dense, sparse = _dense(), _sparse()

    peak_gap = abs(_dbfs(normalize_wav(dense, "peak"))
                   - _dbfs(normalize_wav(sparse, "peak")))
    rms_gap = abs(_dbfs(normalize_wav(dense, "rms"))
                  - _dbfs(normalize_wav(sparse, "rms")))

    assert peak_gap > 10, (
        f"synthetic fixtures are not far enough apart ({peak_gap:.1f} dB)"
    )
    assert rms_gap < peak_gap / 2
    assert rms_gap <= RMS_MAX_BOOST_DB + 1


def test_rms_mode_caps_the_boost():
    """A very sparse track should end up a little quiet, not crushed flat."""
    sparse = _sparse(hits=1)
    peaked = _dbfs(normalize_wav(sparse, "peak"))
    boost = _dbfs(normalize_wav(sparse, "rms")) - peaked
    assert 0 < boost <= RMS_MAX_BOOST_DB + 0.5


def test_rms_mode_turns_loud_tracks_down():
    hot = _dense(amplitude=0.95)
    assert _dbfs(hot) > RMS_TARGET_DBFS  # fixture really is above target
    assert _dbfs(normalize_wav(hot, "rms")) == pytest.approx(RMS_TARGET_DBFS, abs=0.5)


def test_rms_mode_preserves_stereo_balance():
    """Both channels must take the same gain, or the image shifts."""
    wav = torch.cat([_dense(amplitude=0.4), _dense(amplitude=0.2)])
    out = normalize_wav(wav, "rms")
    before = wav[0].abs().max() / wav[1].abs().max()
    after = out[0].abs().max() / out[1].abs().max()
    assert after.item() == pytest.approx(before.item(), rel=1e-3)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="normalize mode"):
        normalize_wav(_dense(), "loudness-war")
