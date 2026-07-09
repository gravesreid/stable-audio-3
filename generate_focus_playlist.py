"""Generate a focus/concentration playlist with Stable Audio 3.

Example:
    uv run python generate_focus_playlist.py --hours 1.5 --out ~/music/focus
"""

import argparse
import random

from playlist_common import add_common_args, generate_playlist, n_tracks_for

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


def main(args):
    n_tracks = n_tracks_for(args)

    # Cycle through the prompt bank so every style is used before any repeats.
    rng = random.Random(args.seed if args.seed >= 0 else None)
    playlist = []
    while len(playlist) < n_tracks:
        batch = PROMPTS[:]
        rng.shuffle(batch)
        playlist.extend(batch)
    playlist = playlist[:n_tracks]

    generate_playlist(args, playlist, NEGATIVE_PROMPT, rng)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate a focus-music playlist")
    add_common_args(p, default_out="focus_playlist")
    main(p.parse_args())
