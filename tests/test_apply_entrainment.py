"""Tests for the gamma-band modulation in apply_entrainment.py.

The two things that matter are that the modulation actually lands at the rate
asked for, and that it stays out of the bass -- the whole reason for the band
split is that a 40 Hz sideband on a 60 Hz kick is a 20 Hz ghost nobody hears and
everybody's headroom pays for. Both are checked on synthetic signals: a high
tone stands in for the band that should be modulated, a low one for the band
that should come through untouched.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apply_entrainment import GAMMA_HZ, limit_to_peak, modulate  # noqa: E402

SR = 44100


def _tone(hz: float, seconds: float = 2.0, amplitude: float = 0.5):
    t = torch.arange(int(SR * seconds)) / SR
    return (amplitude * torch.sin(2 * math.pi * hz * t)).unsqueeze(0)


def _envelope_at(wav, hz: float) -> float:
    """Magnitude of the `hz` component of the signal's envelope, rel. to DC.

    Rectifying and taking the spectrum is a crude envelope follower, but it is
    the same thing the ear does first, so a peak here is the modulation a
    listener would perceive.
    """
    env = wav.abs().mean(dim=0)
    spec = torch.fft.rfft(env * torch.hann_window(env.shape[-1])).abs()
    freqs = torch.fft.rfftfreq(env.shape[-1], 1 / SR)
    band = (freqs > hz - 1) & (freqs < hz + 1)
    return (spec[band].max() / spec[0]).item()


def test_depth_zero_is_identity():
    """The band split must reconstruct exactly, or every track gets a phase notch."""
    wav = _tone(2000) + _tone(80)
    assert torch.allclose(modulate(wav, SR, depth=0.0), wav, atol=1e-6)


def test_modulation_lands_at_the_requested_rate():
    wav = _tone(2000)
    out = modulate(wav, SR, hz=GAMMA_HZ, depth=0.3)
    assert _envelope_at(out, GAMMA_HZ) > 10 * _envelope_at(wav, GAMMA_HZ)


def test_deeper_modulation_is_stronger():
    wav = _tone(2000)
    shallow = _envelope_at(modulate(wav, SR, depth=0.1), GAMMA_HZ)
    deep = _envelope_at(modulate(wav, SR, depth=0.4), GAMMA_HZ)
    assert deep > shallow


def test_bass_is_left_alone():
    """A 60 Hz kick must not pick up the 20 Hz sideband that whole-mix AM gives it."""
    wav = _tone(60)
    out = modulate(wav, SR, depth=0.5, crossover=1000.0)
    assert _envelope_at(out, GAMMA_HZ) < 0.02


@pytest.mark.parametrize("depth", [0.1, 0.3, 0.6, 1.0])
def test_never_exceeds_input_peak(depth):
    """Tracks arrive peak-normalized to the ceiling; modulation must not clip them.

    Ducking the high band nearly guarantees this on its own, so these signals
    mostly pass without the guard in modulate -- the case that needs it is rarer
    than any tone or noise reproduces (see test_limit_to_peak_*). They are here
    to pin the invariant, not to exercise the guard.
    """
    torch.manual_seed(0)
    signals = {
        "tones": _tone(2000, amplitude=0.7) + _tone(120, amplitude=0.4),
        "noise": torch.randn(1, SR) * 0.2,
        "clipped": (1.6 * _tone(110) + 0.5 * _tone(2500)).clamp(-0.95, 0.95),
    }
    for name, wav in signals.items():
        wav = wav / wav.abs().max() * 0.95
        out = modulate(wav, SR, depth=depth)
        assert out.abs().max() <= wav.abs().max() + 1e-6, name


def test_limit_to_peak_scales_down_when_over():
    """The guard that catches the ringing case modulate cannot avoid."""
    wav = _tone(1000, amplitude=0.99)
    out = limit_to_peak(wav, 0.95)
    assert out.abs().max() == pytest.approx(0.95, abs=1e-6)


def test_limit_to_peak_leaves_quieter_audio_alone():
    """One-sided: a track ducked below its old peak must not be boosted back."""
    wav = _tone(1000, amplitude=0.80)
    assert torch.equal(limit_to_peak(wav, 0.95), wav)


def test_stereo_channels_stay_aligned():
    """Both channels share one carrier -- a phase split would smear the image."""
    left, right = _tone(2000), _tone(3000)
    wav = torch.cat([left, right], dim=0)
    out = modulate(wav, SR, depth=0.4)
    separate = torch.cat([modulate(left, SR, depth=0.4),
                          modulate(right, SR, depth=0.4)], dim=0)
    assert torch.allclose(out, separate, atol=1e-6)


@pytest.mark.parametrize("depth", [-0.1, 1.5])
def test_rejects_out_of_range_depth(depth):
    with pytest.raises(ValueError):
        modulate(_tone(2000), SR, depth=depth)
