"""Prompt bank for the mellow playlist.

Same combinatorial engine as the focus and lifting banks (see prompt_bank.py):
each family is a sub-style with its own slot pools, and a prompt draws one value
from each pool, so a short hand-written family expands into thousands of distinct
prompts and a long session never repeats.

Where this bank differs from the other two is the *shape* of its prompts, which
follows the two prompting guides rather than the free prose the older banks use:

1. Labelled metadata fields, in the guide's order -- TrackType, VocalType,
   Format, Genre, Subgenre, Instruments, Moods, Tempo/BPM -- then one prose
   sentence of detail. Stable Audio 3 was trained on Freesound and AudioSparx
   audio *and their metadata*, so prompts shaped like that metadata land closer
   to the training distribution. `TrackType: Music, VocalType: Instrumental` is
   called out in docs/guides/prompting.md as producing higher quality, more
   semantically coherent output, and the single-instrument families lead with
   `TrackType: Instrument` to keep the arrangement from filling in a band.

2. No negations anywhere. The older banks say "no vocals", "no percussion", "no
   drums" in the positive prompt, which asks a text encoder to represent absence
   -- it has no reliable way to do that, and naming drums at all makes drums more
   likely, not less. Everything here is stated positively instead:
   `VocalType: Instrumental` carries the no-vocals requirement, `TrackType:
   Instrument` with `Format: Solo` carries "no band", and `Tempo: Slow` with a
   beatless Genre carries "no beat". The negative prompt still lists what to
   avoid, but it only takes effect above `--cfg-scale 1`.

The job is unchanged: music to destress to and to leave running as background,
that never asks for your attention. Slow, soft and warm throughout -- felt piano,
chamber strings, ambient pads, downtempo, ambient dub, plucked acoustic
instruments, drones. The string-crossover families are the same instruments as
the focus bank, played quietly. Nothing loud, sharp or eventful gets into a slot
value, because one hard transient is what pulls your ear back off whatever you
were doing.
"""

from prompt_bank import Family
from prompt_bank import build_prompts as _build_prompts
from prompt_bank import capacity as _capacity

# Field order is the one both guides give, and both say the order matters. The
# pools are sized 7/4/8/5/6 (+11 for bpm) -- coprime-ish, so cycling them
# odometer-style walks lcm(7,4,8,5,6) = 840 combinations per beatless family
# before one comes back around, far more than any session will ask for.
_FIELDS = (
    "TrackType: {track_type}, VocalType: Instrumental{fmt} | "
    "Genre: {genre} | Subgenre: {subgenre} | Instruments: {instruments} | "
    "Moods: {moods} | {tempo}. {detail}"
)


def _template(track_type: str, tempo: str, fmt: str = "") -> str:
    """One family's prompt shape. `tempo` is "Tempo: Slow" or "BPM: {bpm}"."""
    return _FIELDS.format(
        track_type=track_type,
        fmt=f", Format: {fmt}" if fmt else "",
        genre="{genre}",
        subgenre="{subgenre}",
        instruments="{instruments}",
        moods="{moods}",
        tempo=tempo,
        detail="{detail}",
    )


# Single instruments get TrackType: Instrument (the guide's lever for an isolated
# part); produced/ensemble material gets TrackType: Music.
SOLO = _template("Instrument", "Tempo: Medium", fmt="Solo")
ENSEMBLE = _template("Music", "Tempo: Medium", fmt="Orchestra")
AMBIENT = _template("Music", "Tempo: Medium")
WITH_BPM = _template("Music", "BPM: {bpm}")

FAMILIES = [
    # --------------------------------------------------------------- felt piano
    Family(
        "felt_piano", 3.0, SOLO,
        {
            "genre": [
                "Neoclassical", "Classical", "Ambient", "Contemporary Classical",
                "Minimalism", "New Age", "Cinematic",
            ],
            "subgenre": [
                "Solo Piano", "Ambient Piano", "Piano Miniature",
                "Slow Minimal Piano",
            ],
            "instruments": [
                "felt piano",
                "muted upright piano",
                "close-miked grand piano with soft dampers",
                "prepared piano, felt over the strings",
                "warm upright piano",
                "grand piano played una corda",
                "old parlour piano, mellow tone",
                "grand piano with a faint synth pad underneath",
            ],
            "moods": [
                "Calm, Peaceful, Reflective",
                "Tender, Intimate, Still",
                "Spacious, Atmospheric, Serene",
                "Wistful, Gentle, Unhurried",
                "Warm, Soothing, Contemplative",
            ],
            # These stay continuous on purpose. Asking for "long pauses" or
            # "sparse" here got tracks that were 30-60% near-silence over 120s,
            # which reads as the music having stopped -- the arrangement should be
            # unhurried, but it has to keep playing.
            "detail": [
                "A simple repeating right-hand melody with the sustain pedal held "
                "down throughout.",
                "Slow arpeggios circling the middle register continuously, with a "
                "long natural reverb tail.",
                "Sustained chords flowing one into the next, the felt softening "
                "every attack.",
                "A quiet melody in the upper register over slow left-hand pedal "
                "tones.",
                "A flowing melody over held chords, recorded close in a warm "
                "wooden room with faint key noise and tape hiss.",
                "Soft rubato phrasing that stays inside one hushed dynamic range "
                "from beginning to end.",
            ],
        },
    ),
    # ---------------------------------------------------------- chamber strings
    Family(
        "chamber_strings", 2.5, ENSEMBLE,
        {
            "genre": [
                "Neoclassical", "Classical", "Chamber Music",
                "Contemporary Classical", "Cinematic", "Ambient", "Minimalism",
            ],
            "subgenre": [
                "Slow Chamber Strings", "String Adagio", "Ambient Strings",
                "Minimal String Ensemble",
            ],
            "instruments": [
                "cello, viola, violin",
                "solo cello with a soft string section",
                "two cellos in close harmony",
                "string quartet",
                "cello, double bass, sustained upper strings",
                "viola and cello, warm and low",
                "string ensemble with a faint synth pad",
                "solo violin over sustained strings",
            ],
            "moods": [
                "Calm, Serene, Spacious",
                "Tender, Melancholic, Warm",
                "Reflective, Peaceful, Still",
                "Soothing, Intimate, Unhurried",
                "Atmospheric, Gentle, Contemplative",
            ],
            "detail": [
                "Long bowed swells with soft, even dynamics from start to finish.",
                "Slow sustained lines in a warm hall, plenty of air around the "
                "ensemble.",
                "A cello melody moving in slow steps beneath held upper strings.",
                "Occasional pizzicato punctuating long sustained chords.",
                "Sustained lines close-miked, faint rosin texture, unhurried "
                "phrasing.",
                "Slow harmonic changes, each chord blooming into the next.",
            ],
        },
    ),
    # ---------------------------------------------------------- ambient synths
    Family(
        "ambient_pads", 3.0, AMBIENT,
        {
            "genre": [
                "Ambient", "Electronic", "New Age", "Cinematic", "Experimental",
                "Minimalism", "Chillout",
            ],
            "subgenre": [
                "Ambient Drift", "Modular Ambient", "Tape Ambient",
                "Slow Ambient Electronica",
            ],
            "instruments": [
                "warm analog synthesizer pads",
                "modular synthesizer with slow filter sweeps",
                "Juno-style pads and soft sine tones",
                "mellotron pad and tape-warped strings",
                "granular synthesizer textures",
                "soft bell synth over a deep sub layer",
                "Rhodes electric piano through long reverb",
                "wavetable pads, slowly evolving",
            ],
            "moods": [
                "Spacious, Atmospheric, Ethereal",
                "Calm, Serene, Weightless",
                "Warm, Hazy, Dreamlike",
                "Peaceful, Soothing, Still",
                "Reflective, Distant, Gentle",
            ],
            "detail": [
                "The chords change every eight bars and everything drifts through "
                "long reverb.",
                "Slow tidal swells, one texture dissolving into the next.",
                "Tape saturation and a faint noise floor beneath sustained chords.",
                "A single melodic line surfaces and fades back into the pad.",
                "Long tape delay smears the harmony into a wash.",
                "Beatless and sustained, the filter moving slowly across the pad.",
            ],
        },
    ),
    # -------------------------------------------------------------- downtempo
    Family(
        "downtempo", 2.0, WITH_BPM,
        {
            "genre": [
                "Downtempo", "Chillout", "Trip Hop", "Lo-Fi Hip Hop", "Electronica",
                "Jazz", "Ambient",
            ],
            "subgenre": [
                "Lo-Fi Chillhop", "Slow Trip Hop", "Organic Downtempo",
                "Ambient Beats",
            ],
            "instruments": [
                "Rhodes electric piano, upright bass, brushed drums",
                "vibraphone, soft kick, warm sub bass",
                "muted electric guitar, dusty drum machine, sine bass",
                "felt piano, brushed snare, double bass",
                "muted trumpet, Rhodes, softly swung drums",
                "warm analog synth, tape drums, round bass",
                "harp, soft percussion, deep sub bass",
                "electric piano, shaker, mellow sub bass",
            ],
            "moods": [
                "Calm, Nostalgic, Warm",
                "Mellow, Hazy, Peaceful",
                "Soothing, Laid-Back, Spacious",
                "Reflective, Dusty, Gentle",
                "Serene, Soulful, Unhurried",
            ],
            "detail": [
                "Brushed drums sit low in the mix and the dynamics stay even "
                "throughout.",
                "Vinyl crackle and warm tape saturation over a loose swung groove.",
                "Sustained chords over a slow walking bass, the drums kept quiet.",
                "Muffled low-passed drums under one simple repeating figure.",
                "Warm round bass, brushes on the snare, an easy loping feel.",
                "Recorded dusty and close, one groove holding for the whole track.",
            ],
            "bpm": ["68", "70", "72", "74", "75", "76", "78", "80", "82", "84", "86"],
        },
    ),
    # ------------------------------------------------------------- ambient dub
    Family(
        "ambient_dub", 2.0, WITH_BPM,
        {
            "genre": [
                "Dub Techno", "Ambient Techno", "Deep House", "Dub", "Electronic",
                "Minimal Techno", "Chillout",
            ],
            "subgenre": [
                "Deep Dub Techno", "Ambient Dub", "Slow Deep House",
                "Minimal Ambient Techno",
            ],
            "instruments": [
                "muffled chord pads, tape echo, soft kick",
                "deep sub bass, filtered chords, brushed hats",
                "marimba, warm bass, quiet drum machine",
                "analog pads through spring reverb, soft kick",
                "Rhodes chords, dub delay, round bass",
                "sine bass, muffled chord swells, soft hats",
                "warm organ chords, tape delay, gentle kick",
                "filtered pads, deep bass, soft rimshot",
            ],
            "moods": [
                "Calm, Hypnotic, Spacious",
                "Deep, Warm, Soothing",
                "Atmospheric, Cavernous, Serene",
                "Peaceful, Hazy, Unhurried",
                "Reflective, Submerged, Gentle",
            ],
            "detail": [
                "One chord repeats on the offbeat and echoes far into the reverb.",
                "A soft muffled kick holds the pulse and everything is low-passed.",
                "Chords wash in and out through long tape delay.",
                "Deep sub bass moves in slow steps under a quiet groove.",
                "Hypnotic and unchanging, the hats brushed and low in the mix.",
                "One chord cycling in a warm analog haze, the dub delay feeding "
                "back gently.",
            ],
            "bpm": ["88", "90", "92", "95", "96", "98", "100", "104", "108", "110",
                    "112"],
        },
    ),
    # -------------------------------------------------------- plucked acoustic
    Family(
        "plucked", 1.5, SOLO,
        {
            "genre": [
                "Folk", "Classical", "Neoclassical", "Acoustic", "New Age",
                "Ambient", "World",
            ],
            # Instrument-agnostic on purpose: the instruments pool ranges from
            # guitar to music box, and a subgenre naming one of them would
            # contradict the Instruments field seven times out of eight.
            "subgenre": [
                "Slow Fingerstyle", "Ambient Acoustic", "Flowing Acoustic Miniature",
                "Minimal Plucked Instrumental",
            ],
            "instruments": [
                "nylon-string classical guitar",
                "fingerpicked steel-string acoustic guitar",
                "concert harp",
                "kalimba",
                "music box",
                "celeste and glockenspiel",
                "hammered dulcimer",
                "acoustic guitar over a soft pad",
            ],
            "moods": [
                "Calm, Tender, Intimate",
                "Peaceful, Delicate, Warm",
                "Serene, Wistful, Spacious",
                "Soothing, Nostalgic, Still",
                "Reflective, Gentle, Unhurried",
            ],
            "detail": [
                "A slow fingerpicked pattern repeating continuously, evenly spaced.",
                "A plucked melody ringing out into a warm room, each phrase "
                "following the last.",
                "Rippling arpeggios in the middle register, unhurried and even.",
                "A steady picked figure, close-miked with soft fret noise, "
                "intimate and dry.",
                "Open strings ringing into one another, the sound continuous.",
                "One simple figure cycling with small variations, dynamics hushed.",
            ],
        },
    ),
    # -------------------------------------------------------------- warm drone
    Family(
        "drone", 1.5, AMBIENT,
        {
            "genre": [
                "Ambient", "Drone", "Experimental", "New Age", "Electronic",
                "Cinematic", "Minimalism",
            ],
            "subgenre": [
                "Tape Drone", "Harmonic Drone", "Sustained Drone",
                "Deep Ambient Drone",
            ],
            "instruments": [
                "bowed cello drone and low synthesizer",
                "sustained analog synthesizer tones",
                "tape loops of warm strings",
                "harmonium and soft sine tones",
                "bowed guitar over a deep sub layer",
                "struck bowls and a low pad",
                "processed field recordings and a warm pad",
                "a deep sub tone with soft upper harmonics",
            ],
            "moods": [
                "Still, Deep, Meditative",
                "Calm, Weightless, Spacious",
                "Warm, Enveloping, Serene",
                "Peaceful, Vast, Soothing",
                "Atmospheric, Distant, Hushed",
            ],
            "detail": [
                "One sustained tone slowly changes color across the whole piece.",
                "Faint rain and distant water under a low sustained chord.",
                "Soft harmonics rise out of the drone and settle back into it.",
                "Slow beating overtones, the volume swelling and receding gently.",
                "One low sustained chord under a long reverb tail and faint tape "
                "hiss.",
                "Deep and warm, holding a single chord for the full duration.",
            ],
        },
    ),
]

# Only reachable above --cfg-scale 1. The positive prompt does the real work here
# (VocalType: Instrumental, Tempo: Slow, calm Moods); this is the backstop, and it
# is the one place negation belongs.
NEGATIVE_PROMPT = (
    "vocals, singing, choir, vocal chops, spoken word, lyrics, "
    "aggressive, distorted, harsh, loud, shrill, pounding kick, heavy drums, "
    "fast tempo, high energy, uptempo, big drops, build-up, "
    "sudden dynamic changes, sudden silence, glitchy, clipping, low quality"
)


def capacity() -> int:
    """Longest mellow playlist the bank can fill without repeating a prompt."""
    return _capacity(FAMILIES)


def build_prompts(n: int, rng) -> tuple:
    """Return (prompts, per-family counts): n distinct prompts, families interleaved."""
    return _build_prompts(n, FAMILIES, rng)
