"""Generate a focus/concentration playlist with Stable Audio 3.

Sized for a full working day: at the default 8 hours every one of the 240 tracks
gets its own prompt, so nothing comes back around. The prompt bank lives in
focus_prompts.py.

Example:
    uv run python generate_focus_playlist.py
    uv run python generate_focus_playlist.py --hours 2 --out ~/music/focus
"""

import argparse
import random

from focus_prompts import NEGATIVE_PROMPT, build_prompts, capacity
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
        print(f"  {name:18s} {n:3d} tracks")

    generate_playlist(args, playlist, NEGATIVE_PROMPT, rng)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate a focus-music playlist")
    add_common_args(p, default_out="focus_playlist", default_hours=8.0)
    main(p.parse_args())
