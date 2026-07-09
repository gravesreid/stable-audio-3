"""Generate a focus/concentration playlist with Stable Audio 3.

Example:
    uv run python generate_focus_playlist.py --hours 1.5 --out ~/music/focus
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torchaudio

from stable_audio_3 import StableAudioModel

# Each prompt is a distinct sub-style so the playlist doesn't drift into one texture.
# Steady tempo, minimal vocals, no big dynamic swings -- the things that make music
# recede into the background instead of grabbing attention.
PROMPTS = [
    # --- driving / momentum ---
    "Coding electronic music, steady four-on-the-floor kick, warm analog bass, hypnotic arpeggio, no vocals, 120 BPM",
    "Minimal techno for deep focus, dry rimshot groove, subtle acid line, evolving filter sweeps, no vocals, 126 BPM",
    "Melodic house for concentration, soft plucked synth motif, rolling bassline, gentle sidechain, no vocals, 122 BPM",
    "Progressive house study music, long pad swells, muted arpeggios, steady kick, hypnotic and unobtrusive, no vocals, 124 BPM",
    "Dub techno, deep chord stabs drenched in reverb, tape delay, subterranean bass, no vocals, 122 BPM",
    "Detroit-influenced electro, crisp drum machine, warm strings, restrained melody, no vocals, 128 BPM",
    "Microhouse, clicky percussion, tiny vinyl crackle, understated bassline, hypnotic loop, no vocals, 120 BPM",
    "Deep house for working, muffled organ chords, soft shuffled hats, warm sub bass, no vocals, 121 BPM",
    "Tech house groove for productivity, tight percussion, hypnotic bass riff, minimal changes, no vocals, 125 BPM",
    "Trance-influenced focus music, gentle rolling bass, shimmering arpeggio, wide pads, no vocals, 130 BPM",
    # --- ambient / calm ---
    "Ambient techno, distant pads, soft kick pulse, granular textures, slowly evolving, no vocals, 118 BPM",
    "Ambient electronic focus music, warm evolving synth pads, gentle bell tones, no percussion, no vocals",
    "Berlin school sequencer music, hypnotic analog sequence, slow filter movement, spacious, no vocals, 110 BPM",
    "Downtempo electronica, soft dusty drums, mellow Rhodes chords, warm tape saturation, no vocals, 95 BPM",
    "Chillwave study music, hazy synth chords, gentle groove, nostalgic and soft, no vocals, 100 BPM",
    "Ambient dub, deep echoing chords, slow pulse, oceanic reverb, meditative, no vocals, 90 BPM",
    "Generative ambient music, overlapping sine tones, slow harmonic drift, no drums, no vocals",
    "Kosmische synthesizer music, steady analog pulse, warm drifting pads, meditative, no vocals, 112 BPM",
    # --- rhythmic / textural ---
    "Lo-fi house, crunchy drums, warm detuned chords, tape hiss, relaxed swing, no vocals, 118 BPM",
    "Glitchy IDM for concentration, intricate but soft percussion, warm melodic pads, no vocals, 110 BPM",
    "Braindance electronica, playful melodic synths, crisp broken beat, gentle mood, no vocals, 115 BPM",
    "Liquid drum and bass for focus, rolling breakbeat, deep sub bass, lush jazzy chords, no vocals, 174 BPM",
    "Atmospheric drum and bass, restrained amen break, cavernous pads, deep bass, no vocals, 172 BPM",
    "Downtempo breakbeat, dusty chopped drums, warm bass, soft atmospheric pads, no vocals, 92 BPM",
    "Modular synth patch, self-generating polyrhythmic patterns, clean digital timbres, no vocals, 120 BPM",
    "Minimal electronic music, repetitive marimba-like synth pattern, soft kick, phase-shifting, no vocals, 116 BPM",
    # --- hybrid / organic ---
    "Neo-classical electronic, arpeggiated piano, soft synth pad underneath, quiet steady pulse, no vocals, 100 BPM",
    "Electronic music with warm analog strings, gentle pizzicato sequence, soft kick, focused and calm, no vocals, 114 BPM",
    "Nordic electronic music, icy synth textures, sparse percussion, wide reverb, contemplative, no vocals, 105 BPM",
    "Jazzy electronic groove for studying, brushed drum machine, warm electric piano chords, walking sub bass, no vocals, 108 BPM",
    "Organic house, hand percussion, earthy bass, soft marimba melody, hypnotic, no vocals, 118 BPM",
    "Synthwave for focus, steady pulsing bass, clean arpeggio, restrained melody, no vocals, 110 BPM",
    "Electronic music with binaural-style drone underneath, steady pulse, minimal melody, no vocals, 120 BPM",
    "Cinematic electronic underscore, slow-building pads, soft ticking percussion, no vocals, 100 BPM",
]

NEGATIVE_PROMPT = (
    "vocals, singing, speech, harsh distortion, sudden loud transitions, "
    "dissonance, applause, crowd noise, low quality"
)


def slugify(text: str, max_len: int = 48) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:max_len].strip("-")


def main(args):
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_tracks = max(1, round(args.hours * 3600 / args.duration))

    # Cycle through the prompt bank so every style is used before any repeats.
    rng = random.Random(args.seed if args.seed >= 0 else None)
    playlist = []
    while len(playlist) < n_tracks:
        batch = PROMPTS[:]
        rng.shuffle(batch)
        playlist.extend(batch)
    playlist = playlist[:n_tracks]

    print(f"Generating {n_tracks} x {args.duration}s tracks "
          f"(~{n_tracks * args.duration / 3600:.2f} hours) into {out_dir}")

    model = StableAudioModel.from_pretrained(args.model, model_half=args.model_half)
    sample_rate = model.model.sample_rate

    manifest = []
    for i, prompt in enumerate(playlist, start=1):
        seed = rng.randrange(2**31 - 1)
        torch.manual_seed(seed)

        t0 = time.time()
        audio = model.generate(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT if args.cfg_scale > 1.0 else None,
            duration=args.duration,
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

        name = f"{i:03d}-{slugify(prompt)}.wav"
        path = out_dir / name
        torchaudio.save(str(path), wav, sample_rate)

        manifest.append({"file": name, "prompt": prompt, "seed": seed,
                         "duration": args.duration, "steps": args.steps,
                         "cfg_scale": args.cfg_scale})
        print(f"[{i}/{n_tracks}] {elapsed:5.1f}s  {name}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    with (out_dir / "playlist.m3u").open("w") as f:
        f.write("#EXTM3U\n")
        for entry in manifest:
            f.write(f"#EXTINF:{int(entry['duration'])},{entry['prompt'][:60]}\n")
            f.write(f"{entry['file']}\n")

    print(f"\nDone. {len(manifest)} tracks + playlist.m3u in {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate a focus-music playlist")
    p.add_argument("--out", type=str, default="focus_playlist", help="Output directory")
    p.add_argument("--hours", type=float, default=1.0, help="Total playlist length in hours")
    p.add_argument("--duration", type=float, default=120, help="Seconds per track (small-music maxes at 120)")
    p.add_argument("--model", type=str, default="small-music")
    p.add_argument("--model-half", action="store_true", default=True)
    p.add_argument("--steps", type=int, default=8, help="Diffusion steps (8 is the rectified-flow default)")
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="1.0 = the model's default. Raise (e.g. 4) to enable the negative prompt.")
    p.add_argument("--seed", type=int, default=-1, help="Seed for prompt shuffling and per-track seeds; -1 = random")
    main(p.parse_args())
