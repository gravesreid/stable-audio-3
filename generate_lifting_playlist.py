"""Generate a lifting/workout playlist with Stable Audio 3.

Mega-aggressive, four-on-the-floor material to lift heavy shit for reps: no
warmup, no cooldown, just a steady pounding kick from track one. Prompts are
drawn combinatorially from the bank in lifting_prompts.py, so even a long session
never repeats a prompt, and the aggressive families are interleaved so no one
style runs many tracks deep.

Example:
    uv run python generate_lifting_playlist.py
    uv run python generate_lifting_playlist.py --hours 1.25 --out ~/music/lifting
"""

import argparse
import random

from lifting_prompts import NEGATIVE_PROMPT, build_prompts, capacity
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

    generate_playlist(args, playlist, NEGATIVE_PROMPT, rng)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate a lifting-music playlist")
    add_common_args(p, default_out="lifting_playlist", default_hours=1.5)
    main(p.parse_args())
