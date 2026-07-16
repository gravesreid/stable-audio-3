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


def generate_playlist(args, playlist, negative_prompt, rng, phases=None, durations=None):
    """Generate `playlist` (a list of prompt strings) into args.out.

    `phases` is an optional list of the same length labelling each track, used
    for filenames and the m3u titles so the session structure is visible.

    `durations` is an optional per-track length in seconds; defaults to
    args.duration for every track. Use it when phases must hit exact runtimes.
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
        peak = wav.abs().max()
        if peak > 0:
            wav = wav * (0.95 / peak)  # normalize; generations vary a lot in level

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
