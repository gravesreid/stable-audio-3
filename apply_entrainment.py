"""Add a gamma-band amplitude modulation to an already-generated playlist.

Reads a playlist directory, writes modulated copies to a new one, and carries
the manifest and .m3u across so the result is playable as-is. Originals are
never touched.

Example:
    uv run python apply_entrainment.py --in focus_playlist --out focus_playlist_40hz
    uv run python apply_entrainment.py --in focus_playlist --out focus_40 --depth 0.3
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torchaudio

GAMMA_HZ = 40.0
DEPTH = 0.25

# Set by measurement, not by taste. A 1 kHz crossover keeps the sidebands
# maximally out of the way, but on bass-heavy material it also means the
# modulation never reaches the mix envelope: the untouched low band sits ~9 dB
# above the high band on peak-time techno and simply swamps it, measuring +0.0 dB
# of net modulation. Dropping to 150 Hz gets +12 dB of it back while sub-30 Hz
# energy rises only 0.4 dB, against +8.8 dB for modulating the whole mix.
CROSSOVER_HZ = 150.0


def modulate(wav, sample_rate, hz=GAMMA_HZ, depth=DEPTH, crossover=CROSSOVER_HZ):
    """Amplitude-modulate the band above `crossover` at `hz`.

    Modulating the whole mix would be the obvious thing and it is the wrong
    thing. AM puts a copy of every partial `hz` either side of it, so at 40 Hz a
    60 Hz kick throws a 20 Hz ghost -- inaudible, but it eats headroom -- and
    measurably muddies the bottom (+8.8 dB below 30 Hz at depth 0.3). Splitting
    first keeps the sub-bass out of it entirely.

    Where to split is a real tradeoff, not a free win, and `crossover` is worth
    moving per playlist. High settings are safest for the sidebands -- above
    1 kHz a 40 Hz ghost sits a third of a semitone off and vanishes into the
    timbre -- but they also stop the effect reaching the mix envelope on
    bass-heavy material. The default trades some of that safety for an effect
    that actually lands: just above 150 Hz the ghosts are ~4 semitones out, which
    dense material hides and sparse sustained tonal material may not. Raise it
    for anything cello-and-piano shaped.

    The split is `low = lowpass(x); high = x - low`, so the two bands sum back to
    the input exactly, whatever the filter does to phase. At depth 0 this
    function is the identity.

    The carrier swings between `1 - depth` and 1 rather than around 1, so it
    only ever ducks the high band and never boosts it. That is *nearly* enough
    to keep the peak where it was, but not quite: the output is `x - d*high` for
    some d in [0, depth], and where the filter rings the high band runs opposite
    in sign to the mix, so subtracting it nudges the sample up instead of down.
    On a track that arrived peak-normalized to the 0.95 ceiling -- which is
    every track these scripts produce -- there are many samples sitting exactly
    at that ceiling for this to happen at. Measured overshoot on real material
    is small (0.950 -> 0.952) but it is real, so it gets guarded rather than
    argued away.
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {depth}")

    low = torchaudio.functional.lowpass_biquad(wav, sample_rate, crossover)
    high = wav - low

    t = torch.arange(wav.shape[-1], dtype=wav.dtype) / sample_rate
    carrier = 1.0 - (depth / 2.0) * (1.0 - torch.sin(2 * torch.pi * hz * t))
    return limit_to_peak(low + high * carrier, wav.abs().max())


def limit_to_peak(wav, peak):
    """Scale `wav` down if it exceeds `peak`, otherwise leave it alone.

    Scaling to the track's *original* peak rather than to a fixed ceiling is
    what keeps the playlist level-matched: each track comes out exactly as loud
    as it went in, so whatever normalize_wav decided at generation time still
    holds. A one-sided guard, not a normalizer -- a track that ducked below its
    old peak stays ducked, since that is the effect doing its job.
    """
    current = wav.abs().max()
    if current <= peak:
        return wav
    return wav * (peak / current)


def process(in_dir: Path, out_dir: Path, hz: float, depth: float, crossover: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = sorted(in_dir.glob("*.wav"))
    if not tracks:
        raise SystemExit(f"no .wav files in {in_dir}")

    print(f"{len(tracks)} tracks: {hz:g} Hz at depth {depth:g} above {crossover:g} Hz")
    print(f"  {in_dir} -> {out_dir}\n")

    for i, path in enumerate(tracks, start=1):
        wav, sample_rate = torchaudio.load(str(path))
        out = modulate(wav, sample_rate, hz, depth, crossover)
        torchaudio.save(str(out_dir / path.name), out, sample_rate)
        print(f"[{i}/{len(tracks)}] {path.name}  peak "
              f"{wav.abs().max():.3f} -> {out.abs().max():.3f}")

    # Carry the playlist across so the output directory stands on its own, and
    # record what was done to it -- the audio change is not visible in a
    # filename and gets impossible to identify a month later.
    m3u = in_dir / "playlist.m3u"
    if m3u.exists():
        shutil.copy(m3u, out_dir / "playlist.m3u")

    manifest_path = in_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest:
            entry["entrainment"] = {"hz": hz, "depth": depth, "crossover": crossover}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nDone. {len(tracks)} tracks in {out_dir}")


def main(args):
    process(Path(args.in_dir).expanduser(), Path(args.out).expanduser(),
            args.hz, args.depth, args.crossover)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Add a gamma-band AM to a playlist")
    p.add_argument("--in", dest="in_dir", required=True, help="Input playlist directory")
    p.add_argument("--out", required=True, help="Output directory (created)")
    p.add_argument("--hz", type=float, default=GAMMA_HZ,
                   help="Modulation rate; 40 is the gamma-band target")
    p.add_argument("--depth", type=float, default=DEPTH,
                   help="0 = no change, 1 = full duck. 0.2 is audible but not buzzy.")
    p.add_argument("--crossover", type=float, default=CROSSOVER_HZ,
                   help="Only the band above this is modulated")
    main(p.parse_args())
