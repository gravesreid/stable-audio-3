"""Generate a mellow/destress playlist with Stable Audio 3.

Background music to leave running: slow, soft and even, with no build-ups, no
drops and nothing that asks for your attention. Prompts are drawn
combinatorially from the bank in mellow_prompts.py, so a long session never
repeats a prompt, and the families are interleaved so the playlist keeps drifting
between piano, strings, pads and drones instead of sitting in one texture for
half an hour.

There are no phases. A wind-down arc would mean the playlist gets somewhere,
which is the opposite of what background noise is for -- every track is equally
mellow, so it does not matter where you drop in or how long you leave it on.

Example:
    uv run python generate_mellow_playlist.py
    uv run python generate_mellow_playlist.py --hours 3 --out ~/music/mellow
    uv run python generate_mellow_playlist.py --cfg-scale 4  # enforce the negatives
"""

import argparse
import random

from mellow_prompts import NEGATIVE_PROMPT, build_prompts, capacity
from playlist_common import add_common_args, generate_playlist, n_tracks_for


def main(args):
    n_tracks = n_tracks_for(args)
    if n_tracks > capacity():
        max_hours = capacity() * args.duration / 3600
        raise SystemExit(
            f"--hours {args.hours:g} needs {n_tracks} distinct prompts, but the "
            f"bank tops out at {capacity()} ({max_hours:.0f} hours at "
            f"{args.duration:g}s per track)"
        )

    rng = random.Random(args.seed if args.seed >= 0 else None)
    playlist, counts = build_prompts(n_tracks, rng)

    for name, n in counts.items():
        print(f"  {name:14s} {n:3d} tracks")

    # RMS rather than peak normalization: peak-normalized, a sparse piano track
    # lands ~20 dB quieter in the body than a dense pad, and a level jump that
    # size is the one thing background music must never do.
    generate_playlist(args, playlist, NEGATIVE_PROMPT, rng, normalize="rms")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate a mellow destress playlist")
    add_common_args(p, default_out="mellow_playlist", default_hours=2.0)
    main(p.parse_args())
