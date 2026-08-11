"""Shared generation machinery for the playlist scripts."""

import json
import time
from pathlib import Path

import torch
import torchaudio

from stable_audio_3 import StableAudioModel


def slugify(text: str, max_len: int = 48) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:max_len].strip("-")


# Peak normalization is right for the workout playlists: every track hits the
# same ceiling, and dense aggressive material has a small crest factor anyway. It
# is wrong for mellow material. A sparse piano piece normalized on its loudest
# attack sits ~20 dB quieter in the body than a dense pad does, and a background
# playlist that jumps 20 dB between tracks is a background playlist you look up
# from. RMS mode matches tracks by body instead: it aims for a loudness target,
# caps how far it will push a track past peak normalization (so a very sparse
# track ends up a little quiet rather than crushed), and rounds off whatever pokes
# past the knee instead of clipping it.
RMS_TARGET_DBFS = -20.0
RMS_MAX_BOOST_DB = 8.0
SOFT_CLIP_KNEE = 0.6
CEILING = 0.95


def soft_clip(wav, knee: float = SOFT_CLIP_KNEE, ceiling: float = CEILING):
    """Squash |wav| above `knee` into [knee, ceiling), smoothly and monotonically.

    tanh is used only above the knee, so it starts with unity slope there: quiet
    passages pass through untouched and only the peak tips are rounded.
    """
    room = ceiling - knee
    over = (wav.abs() - knee).clamp(min=0)
    magnitude = wav.abs().clamp(max=knee) + room * torch.tanh(over / room)
    return torch.sign(wav) * magnitude


def normalize_wav(wav, mode: str = "peak"):
    """Scale `wav` to a consistent level. mode is "peak" or "rms"."""
    peak = wav.abs().max()
    if peak <= 0:
        return wav

    peak_gain = CEILING / peak
    if mode == "peak":
        return wav * peak_gain
    if mode != "rms":
        raise ValueError(f"unknown normalize mode {mode!r}")

    rms = wav.pow(2).mean().sqrt()
    gain = 10 ** (RMS_TARGET_DBFS / 20) / rms
    gain = min(gain, peak_gain * 10 ** (RMS_MAX_BOOST_DB / 20))
    return soft_clip(wav * gain)


def add_model_args(parser, default_out: str):
    """Output, model, and sampling flags. Nothing about playlist length."""
    parser.add_argument("--out", type=str, default=default_out, help="Output directory")
    parser.add_argument("--duration", type=float, default=120,
                        help="Seconds per track (small-music maxes at 120)")
    parser.add_argument("--model", type=str, default="small-music")
    parser.add_argument("--model-half", action="store_true", default=True)
    parser.add_argument("--steps", type=int, default=8,
                        help="Diffusion steps (8 is the rectified-flow default)")
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="1.0 = the model's default. Raise (e.g. 4) to enable the negative prompt.")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Seed for track ordering and per-track seeds; -1 = random")
    return parser


def add_common_args(parser, default_out: str, default_hours: float = 1.0):
    """add_model_args plus --hours, for playlists sized by total runtime."""
    add_model_args(parser, default_out)
    parser.add_argument("--hours", type=float, default=default_hours,
                        help="Total playlist length in hours")
    return parser


def n_tracks_for(args) -> int:
    return max(1, round(args.hours * 3600 / args.duration))


def generate_playlist(args, playlist, negative_prompt, rng, phases=None,
                      durations=None, normalize="peak"):
    """Generate `playlist` (a list of prompt strings) into args.out.

    `phases` is an optional list of the same length labelling each track, used
    for filenames and the m3u titles so the session structure is visible.

    `durations` is an optional per-track length in seconds; defaults to
    args.duration for every track. Use it when phases must hit exact runtimes.

    `normalize` is "peak" or "rms" -- see normalize_wav. Use "rms" when the
    tracks must sit at the same perceived level as each other.
    """
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(playlist)
    if durations is None:
        durations = [args.duration] * total
    assert len(durations) == total, "durations must match playlist length"

    runtime = sum(durations)
    print(f"Generating {total} tracks (~{runtime / 60:.1f} min) into {out_dir}")

    model = StableAudioModel.from_pretrained(args.model, model_half=args.model_half)
    sample_rate = model.model.sample_rate

    manifest = []
    for i, prompt in enumerate(playlist, start=1):
        seed = rng.randrange(2**31 - 1)
        torch.manual_seed(seed)
        duration = durations[i - 1]

        t0 = time.time()
        audio = model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt if args.cfg_scale > 1.0 else None,
            duration=duration,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=seed,
        )
        elapsed = time.time() - t0

        # (batch, channels, samples) -> (channels, samples), float32 in [-1, 1]
        wav = audio[0].cpu()
        wav = normalize_wav(wav, normalize)  # generations vary a lot in level

        phase = phases[i - 1] if phases else None
        stem = slugify(f"{phase}-{prompt}" if phase else prompt)
        name = f"{i:03d}-{stem}.wav"
        torchaudio.save(str(out_dir / name), wav, sample_rate)

        entry = {"file": name, "prompt": prompt, "seed": seed,
                 "duration": round(duration, 2), "steps": args.steps,
                 "cfg_scale": args.cfg_scale}
        if phase:
            entry["phase"] = phase
        manifest.append(entry)
        print(f"[{i}/{total}] {elapsed:5.1f}s  {name}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    with (out_dir / "playlist.m3u").open("w") as f:
        f.write("#EXTM3U\n")
        for entry in manifest:
            title = entry["prompt"][:60]
            if "phase" in entry:
                title = f"[{entry['phase']}] {title}"
            f.write(f"#EXTINF:{int(entry['duration'])},{title}\n")
            f.write(f"{entry['file']}\n")

    print(f"\nDone. {len(manifest)} tracks + playlist.m3u in {out_dir}")
    return manifest
