"""Generate a cardio playlist with Stable Audio 3.

Three fixed-length phases rather than proportional ones: you know how long you
want to warm up, work, and cool down. Track lengths inside a phase are shrunk
evenly so the phase hits its target exactly -- a 5 minute warmup is 3 tracks of
100s, not 3 tracks of 120s overshooting to 6 minutes.

Example:
    uv run python generate_cardio_playlist.py --out ~/music/cardio
    uv run python generate_cardio_playlist.py --warmup 8 --main 40 --cooldown 7
"""

import argparse
import math
import random

from playlist_common import add_model_args, generate_playlist

# Steady tempo matters more here than in lifting. The main phase should hold a
# runnable cadence without the big drops and breakdowns that break your stride.
PHASE_PROMPTS = {
    "warmup": [
        "Uplifting progressive house, warm pads, steady four-on-the-floor kick, gently building, no vocals, 128 BPM",
        "Melodic techno warmup, rolling bassline, bright arpeggio, easy momentum, no vocals, 126 BPM",
        "Deep house with drive, punchy kick, warm sub bass, optimistic chords, no vocals, 124 BPM",
        "Nu-disco groove, elastic bassline, crisp hats, sunny and propulsive, no vocals, 122 BPM",
        "Breaks-influenced warmup, chunky broken beat, warm bass, gathering pace, no vocals, 130 BPM",
        "Tech house, shuffled hats, hypnotic bass riff, steady forward pull, no vocals, 127 BPM",
    ],
    "main": [
        "Driving uplifting trance, rolling bassline, soaring arpeggio, relentless forward motion, no vocals, 138 BPM",
        "Peak-time techno for running, tireless kick, hypnotic stabs, steady intensity, no vocals, 140 BPM",
        "Liquid drum and bass, rolling breakbeat, deep sub bass, propulsive and bright, no vocals, 174 BPM",
        "Big room electro house, punchy saw bass, driving snare, high energy, no vocals, 128 BPM",
        "Hard groove techno, tribal percussion, insistent kick, unbroken momentum, no vocals, 136 BPM",
        "Psytrance, rolling triplet bassline, hypnotic acid line, relentless drive, no vocals, 145 BPM",
        "Jump-up drum and bass, punchy breakbeat, bouncing bass riff, energetic, no vocals, 172 BPM",
        "Progressive trance, wide pads, pulsing bassline, euphoric build, no vocals, 134 BPM",
        "Electro house banger, gritty synth bass, sharp claps, unflagging pace, no vocals, 130 BPM",
        "Acid techno, screaming 303 line, steady hard kick, hypnotic and driving, no vocals, 142 BPM",
        "Future rave, dark plucked bass, huge supersaw lead, unyielding energy, no vocals, 132 BPM",
        "Neurofunk drum and bass, snarling reese bass, precise rolling break, no vocals, 174 BPM",
        "Hardstyle instrumental, distorted kick, euphoric lead, enormous drive, no vocals, 150 BPM",
        "Melodic bass house, lurching bassline, bright melodic hook, propulsive, no vocals, 128 BPM",
        "Italo-influenced dance, pulsing octave bass, bright arpeggio, tireless, no vocals, 132 BPM",
        "Breakbeat hardcore, chopped amen break, stabbing bass, rushing energy, no vocals, 160 BPM",
    ],
    "cooldown": [
        "Downtempo electronic comedown, warm pads, slow steady beat, spacious, no vocals, 100 BPM",
        "Ambient techno afterglow, distant pads, slow pulse, reflective, no vocals, 110 BPM",
        "Deep house cooldown, soft chords, gentle rolling bass, easing off, no vocals, 116 BPM",
        "Dub techno, deep echoing chords, tape delay, unwinding, no vocals, 112 BPM",
        "Chillout electronica, mellow Rhodes chords, soft dusty drums, no vocals, 95 BPM",
        "Ambient synth music, warm evolving pads, slow harmonic drift, no drums, no vocals",
    ],
}

NEGATIVE_PROMPT = (
    "vocals, singing, sudden tempo changes, long breakdown, silence, "
    "sparse, hesitant, low energy, low quality"
)


def build_playlist(phase_minutes, max_duration, rng):
    """Return (prompts, phases, durations) hitting each phase's minutes exactly.

    Within a phase, tracks are equal-length and as close to max_duration as
    possible without exceeding it.
    """
    playlist, phases, durations = [], [], []
    for name, minutes in phase_minutes:
        if minutes <= 0:
            continue
        seconds = minutes * 60
        n = math.ceil(seconds / max_duration)
        track_len = seconds / n

        bank = PHASE_PROMPTS[name]
        chosen = []
        while len(chosen) < n:
            batch = bank[:]
            rng.shuffle(batch)
            chosen.extend(batch)

        playlist.extend(chosen[:n])
        phases.extend([name] * n)
        durations.extend([track_len] * n)
    return playlist, phases, durations


def main(args):
    if args.duration > 120 and args.model.startswith("small"):
        raise SystemExit("--duration above 120s is not supported by the small models")

    phase_minutes = [("warmup", args.warmup), ("main", args.main),
                     ("cooldown", args.cooldown)]
    if sum(m for _, m in phase_minutes) <= 0:
        raise SystemExit("Nothing to generate: all phases are zero minutes")

    rng = random.Random(args.seed if args.seed >= 0 else None)
    playlist, phases, durations = build_playlist(phase_minutes, args.duration, rng)

    for name, minutes in phase_minutes:
        n = phases.count(name)
        if n:
            print(f"  {name:9s} {n:2d} tracks x {durations[phases.index(name)]:.0f}s"
                  f" = {minutes:g} min")

    generate_playlist(args, playlist, NEGATIVE_PROMPT, rng,
                      phases=phases, durations=durations)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate a cardio playlist")
    add_model_args(p, default_out="cardio_playlist")
    p.add_argument("--warmup", type=float, default=5, help="Warmup minutes")
    p.add_argument("--main", type=float, default=25, help="Main effort minutes")
    p.add_argument("--cooldown", type=float, default=5, help="Cooldown minutes")
    main(p.parse_args())
