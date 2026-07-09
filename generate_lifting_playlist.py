"""Generate a lifting/workout playlist with Stable Audio 3.

Tracks are ordered as a session arc rather than shuffled uniformly: the warmup
builds, the middle carries your heavy work, and the tail cools down.

Example:
    uv run python generate_lifting_playlist.py --hours 1.25 --out ~/music/lifting
"""

import argparse
import random

from playlist_common import add_common_args, generate_playlist, n_tracks_for

# A lifting session runs long. Prompt banks below are sized so a 90 minute
# playlist draws every track from a distinct prompt.
MIN_HOURS = 1.5

# Phases of a session, each with a share of total runtime and its own prompt bank.
# Weights are relative and get normalized, so they need not sum to 1.
PHASES = [
    (
        "warmup",
        0.12,
        [
            "Deep rolling techno, patient build, warm sub bass, steady kick, no vocals, 124 BPM",
            "Progressive house warmup, wide pads, gradually rising arpeggio, driving kick, no vocals, 126 BPM",
            "Melodic techno, brooding bassline, slowly opening filter, hypnotic momentum, no vocals, 125 BPM",
            "Dark disco, punchy live bass, tight hats, restrained but confident groove, no vocals, 122 BPM",
            "Rolling tech house warmup, crisp shuffled hats, elastic bassline, building, no vocals, 125 BPM",
            "Breaks-influenced warmup, chunky broken beat, warm bass, growing intensity, no vocals, 128 BPM",
        ],
    ),
    (
        "build",
        0.18,
        [
            "Driving peak-time techno, relentless kick, stabbing acid line, rising tension, no vocals, 132 BPM",
            "Big room electro house, distorted saw bass, huge snare build, aggressive, no vocals, 128 BPM",
            "Industrial techno, metallic percussion, pounding kick, menacing drone, no vocals, 138 BPM",
            "Electro-funk workout music, fat synth bass, punchy drum machine, swagger, no vocals, 126 BPM",
            "Acid techno, screaming 303 line, hard kick drum, hypnotic and aggressive, no vocals, 135 BPM",
            "Driving electro-breaks, punchy syncopated drums, snarling bass, no vocals, 130 BPM",
            "Hard groove techno, tribal percussion, insistent kick, mounting pressure, no vocals, 134 BPM",
            "Bass house, lurching wobble bassline, sharp drums, confident swagger, no vocals, 128 BPM",
        ],
    ),
    (
        "peak",
        0.45,
        [
            "Aggressive gym techno, hammering kick, dark distorted bass, adrenaline, no vocals, 140 BPM",
            "Hard dubstep instrumental, colossal wobble bass, crushing drops, heavy, no vocals, 140 BPM",
            "Neurofunk drum and bass, snarling reese bass, precise breakbeat, ferocious, no vocals, 174 BPM",
            "Electronic trap instrumental, booming 808 bass, crisp snappy snares, menacing brass, no vocals, 145 BPM",
            "Hardstyle instrumental, distorted kick, euphoric lead, enormous energy, no vocals, 150 BPM",
            "Heavy electronic rock, distorted guitar riff over pounding synth bass, driving drums, no vocals, 140 BPM",
            "Cinematic battle electronic music, thunderous percussion, brass stabs, roaring synth bass, no vocals, 130 BPM",
            "Breakcore-tinged workout music, frantic breakbeat, huge bass, relentless drive, no vocals, 160 BPM",
            "Aggressive electro-industrial, grinding bass, mechanical percussion, dark, no vocals, 136 BPM",
            "Peak-time hard techno, rumbling kick, siren stabs, hypnotic brutality, no vocals, 145 BPM",
            "Phonk workout music, distorted cowbell melody, heavy 808 bass, aggressive, no vocals, 145 BPM",
            "Epic synth metal instrumental, palm-muted riffing, soaring lead, double kick drums, no vocals, 150 BPM",
            "Aggressive jungle, chopped amen break, deep rumbling bass, raw and urgent, no vocals, 170 BPM",
            "Dark psytrance, rolling triplet bassline, hypnotic acid stabs, relentless, no vocals, 145 BPM",
            "Riddim dubstep instrumental, lurching triplet bass, brutal percussion, no vocals, 140 BPM",
            "Hardcore techno, overdriven kick, blaring stabs, punishing intensity, no vocals, 155 BPM",
            "Drumstep instrumental, half-time monstrous bass drops, huge drums, no vocals, 150 BPM",
            "Aggressive synthwave, distorted arpeggio, pounding gated drums, dark power, no vocals, 140 BPM",
            "Gabber-influenced workout music, distorted kick barrage, raw energy, no vocals, 160 BPM",
            "Cinematic hybrid trailer music, war drums, distorted brass, electronic bass, no vocals, 138 BPM",
        ],
    ),
    (
        "pump",
        0.15,
        [
            "Funky electro house, slap bass synth, punchy claps, confident strut, no vocals, 126 BPM",
            "Big beat, huge breakbeat drums, fuzzy bass, swaggering horn stabs, no vocals, 130 BPM",
            "Nu-disco power groove, driving bassline, bright stabs, propulsive, no vocals, 124 BPM",
            "French house banger, filtered disco loop, thick compression, relentless groove, no vocals, 125 BPM",
            "Breakbeat funk, chunky drums, gritty bass riff, brass hits, swaggering, no vocals, 128 BPM",
            "Ghetto funk, heavy hip-hop break, filthy synth bass, party energy, no vocals, 110 BPM",
            "Electro-boogie, slap synth bass, bright chords, unstoppable strut, no vocals, 118 BPM",
        ],
    ),
    (
        "cooldown",
        0.10,
        [
            "Downtempo electronic comedown, warm pads, slow heavy beat, satisfied and spacious, no vocals, 100 BPM",
            "Deep house cooldown, soft chords, gentle rolling bass, easing off, no vocals, 118 BPM",
            "Ambient techno afterglow, distant pads, slow pulse, reflective, no vocals, 110 BPM",
            "Dub techno cooldown, deep echoing chords, tape delay, unwinding, no vocals, 115 BPM",
            "Downtempo breakbeat comedown, dusty drums, warm bass, spacious, no vocals, 95 BPM",
        ],
    ),
]

# Ballads and ambience are what kill a set. Vocals matter less here than in focus
# music, but chanted hooks still get repetitive on the 40th listen.
NEGATIVE_PROMPT = (
    "vocals, singing, ambient, slow tempo, quiet, gentle, sparse, "
    "acoustic, mellow, low energy, low quality"
)


def build_playlist(n_tracks, rng):
    """Allocate tracks across phases by weight, keeping the session in order."""
    weights = [w for _, w, _ in PHASES]
    total_w = sum(weights)

    # Largest-remainder allocation so the counts sum to exactly n_tracks.
    exact = [n_tracks * w / total_w for w in weights]
    counts = [int(x) for x in exact]
    for idx in sorted(range(len(PHASES)), key=lambda i: exact[i] - counts[i], reverse=True):
        if sum(counts) >= n_tracks:
            break
        counts[idx] += 1

    # Guarantee the peak phase survives even very short sessions.
    if sum(counts) and counts[2] == 0:
        donor = max(range(len(counts)), key=lambda i: counts[i])
        if counts[donor] > 0:
            counts[donor] -= 1
            counts[2] += 1

    playlist, phases = [], []
    for (name, _, prompts), count in zip(PHASES, counts):
        chosen = []
        while len(chosen) < count:
            batch = prompts[:]
            rng.shuffle(batch)
            chosen.extend(batch)
        playlist.extend(chosen[:count])
        phases.extend([name] * count)
    return playlist, phases


def main(args):
    if args.hours < MIN_HOURS and not args.allow_short:
        print(f"Bumping {args.hours}h up to the {MIN_HOURS}h minimum "
              f"(pass --allow-short to override).")
        args.hours = MIN_HOURS

    rng = random.Random(args.seed if args.seed >= 0 else None)
    playlist, phases = build_playlist(n_tracks_for(args), rng)

    for name, _, _ in PHASES:
        n = phases.count(name)
        if n:
            print(f"  {name:9s} {n:2d} tracks ({n * args.duration / 60:.0f} min)")

    generate_playlist(args, playlist, NEGATIVE_PROMPT, rng, phases=phases)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate a lifting-music playlist")
    add_common_args(p, default_out="lifting_playlist", default_hours=MIN_HOURS)
    p.add_argument("--allow-short", action="store_true",
                   help=f"Permit playlists shorter than the {MIN_HOURS}h minimum")
    main(p.parse_args())
