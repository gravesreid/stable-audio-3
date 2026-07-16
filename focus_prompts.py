"""Prompt bank for the focus playlist.

Eight hours at 120s per track is 240 tracks, and a hand-written bank that long
either repeats or drifts into one texture. So prompts are built combinatorially:
each family is a sub-style with its own slot pools, and a prompt is one value
drawn from each pool. Pool sizes are pairwise coprime-ish, so cycling them
odometer-style walks a long path before any combination comes back around --
lcm(5,6,7,8,11) = 9240 prompts per family, which is more than we will ever ask
for.

Families are weighted toward steady, propulsive, cinematic material: driving
electronic and string-led crossover carry the playlist, ambient is a garnish.
Nothing here has vocals, and no slot value mentions voices, choirs or lyrics --
the model will happily sing if invited.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    """A sub-style. `template` is formatted with one value from each slot pool."""

    name: str
    weight: float
    template: str
    slots: dict

    def capacity(self) -> int:
        return math.lcm(*(len(pool) for pool in self.slots.values()))


# Every family below uses this shape, so prompts read consistently to the model:
#   <style>, <lead>, <texture>, <motion>, no vocals, <bpm> BPM
WITH_BPM = "{style}, {lead}, {texture}, {motion}, no vocals, {bpm} BPM"
NO_BPM = "{style}, {lead}, {texture}, {motion}, no vocals"

FAMILIES = [
    # ---------------------------------------------------------------- cinematic
    Family(
        "cinematic", 3.0, WITH_BPM,
        {
            "style": [
                "Cinematic electronic underscore",
                "Hybrid orchestral-electronic score",
                "Widescreen film score electronica",
                "Restrained trailer music, tension held without release",
                "Night drive film score",
                "Sci-fi film score, cold and spacious",
                "Heist montage score, controlled and precise",
            ],
            "lead": [
                "long sustained string ostinato",
                "muted piano motif circling",
                "solo violin tracing a simple line",
                "distant brass held in reserve",
                "glassy synth lead, unhurried",
                "cello line moving in slow steps",
                "plucked harp-like sequence",
                "bowed vibraphone melody",
            ],
            "texture": [
                "warm analog pads underneath",
                "granular string textures in the background",
                "soft tape-saturated strings",
                "wide reverberant space",
                "low drone holding the harmony",
            ],
            "motion": [
                "steady ticking percussion",
                "insistent low pulse",
                "muted heartbeat kick",
                "restrained taiko-like floor toms",
                "staccato strings marking time",
                "sparse rim clicks keeping time",
            ],
            "bpm": ["92", "96", "100", "104", "108", "112", "116", "120", "124",
                    "128", "132"],
        },
    ),
    # ------------------------------------------------- electric violin over EDM
    Family(
        "violin_electronic", 2.5, WITH_BPM,
        {
            "style": [
                "Electronic violin crossover instrumental",
                "Dance instrumental built around a solo violin",
                "Celtic-tinged electronic instrumental, violin lead",
                "Cinematic dance instrumental with a violin lead",
                "Folk-electronic crossover, fiddle over programmed drums",
                "Epic violin electronica",
                "Middle-Eastern-inflected electronic instrumental, violin lead",
            ],
            "lead": [
                "energetic staccato violin riff",
                "soaring legato violin melody",
                "double-stopped violin hook",
                "pizzicato violin ostinato",
                "violin trading phrases with a synth lead",
                "dancing sixteenth-note violin line",
                "violin melody doubled an octave up",
                "keening violin countermelody over a string section",
            ],
            "texture": [
                "layered string section underneath",
                "wide supersaw chords",
                "warm analog pads and plucks",
                "orchestral strings and synth bass together",
                "airy bell textures",
            ],
            "motion": [
                "steady four-on-the-floor kick",
                "punchy programmed drums, no drops",
                "driving syncopated beat",
                "propulsive kick and clap",
                "rolling percussion with hand drums",
                "steady drive, no big breakdowns",
            ],
            "bpm": ["112", "116", "120", "124", "126", "128", "130", "132", "134",
                    "136", "140"],
        },
    ),
    # ------------------------------------------------ layered piano and cellos
    Family(
        "piano_cello", 2.0, WITH_BPM,
        {
            "style": [
                "Cinematic piano and cello duet",
                "Neo-classical crossover for piano and layered cellos",
                "Percussive piano and cello instrumental",
                "Grand piano with a wall of overdubbed cellos",
                "Chamber crossover instrumental, piano and cello",
                "Uplifting piano and cello arrangement, cinematic",
                "Piano and cello score, wide and hopeful",
            ],
            "lead": [
                "rippling piano arpeggios",
                "lyrical cello melody in the upper register",
                "octave piano melody",
                "cello and piano trading the melody",
                "rolling left-hand piano ostinato",
                "cello double-stops carrying the theme",
                "sparse piano melody with lots of air",
                "cascading piano figure over a cello pedal",
            ],
            "texture": [
                "stacked cello harmonies underneath",
                "pizzicato cellos marking the pulse",
                "warm room reverb, close-miked",
                "low cello drone holding the key",
                "subtle synth pad under the acoustic instruments",
            ],
            "motion": [
                "cello body percussion keeping time",
                "steady driving eighth notes",
                "light cajon-like pulse",
                "no drums, momentum from the ostinato",
                "propulsive staccato bass line",
                "gentle rocking rhythm",
            ],
            "bpm": ["84", "88", "92", "96", "100", "104", "108", "112", "116",
                    "120", "126"],
        },
    ),
    # ------------------------------------------------------ driving cello duo
    Family(
        "cello_rock", 1.5, WITH_BPM,
        {
            "style": [
                "Cello rock instrumental, two cellos",
                "Dramatic cello duo instrumental, rock energy",
                "Cello-driven instrumental with a rock rhythm section",
                "Symphonic rock instrumental led by cellos",
                "Aggressive classical crossover for cellos",
                "Cello ensemble instrumental, driving and dark",
                "Baroque-flavored cello rock instrumental",
            ],
            "lead": [
                "distorted cello riff",
                "fast sawing cello ostinato",
                "soaring cello melody over the riff",
                "two cellos trading aggressive lines",
                "cello playing a bass riff with attitude",
                "chugging staccato cello chords",
                "cello melody in the high register, intense",
                "cellos in unison octaves",
            ],
            "texture": [
                "string section swelling behind",
                "gritty amp tone on the low cello",
                "big room reverb",
                "layered cello harmonies",
                "synth bass reinforcing the low end",
            ],
            "motion": [
                "hard-hitting rock drums",
                "driving kick and snare, steady",
                "relentless eighth-note pulse",
                "tight groove, no tempo changes",
                "punchy drums, sustained intensity",
                "propulsive half-time groove",
            ],
            "bpm": ["96", "100", "104", "108", "112", "116", "120", "124", "128",
                    "132", "138"],
        },
    ),
    # --------------------------------------------- steady electronic momentum
    Family(
        "driving", 3.0, WITH_BPM,
        {
            "style": [
                "Melodic techno for deep focus",
                "Progressive house for concentration",
                "Driving electronic coding music",
                "Deep rolling techno for working",
                "Melodic house for concentration",
                "Trance-influenced focus instrumental",
                "Cinematic melodic techno",
            ],
            "lead": [
                "hypnotic arpeggio cycling",
                "plucked synth motif",
                "soft supersaw chord stabs",
                "muted string ostinato over the beat",
                "bright bell sequence",
                "restrained acid line",
                "slowly opening filter on a saw lead",
                "detuned pluck sequence",
            ],
            "texture": [
                "wide evolving pads",
                "warm analog bass underneath",
                "reverb-soaked chord stabs",
                "granular textures in the background",
                "deep sub bass and little else",
            ],
            "motion": [
                "steady four-on-the-floor kick, no drops",
                "rolling bassline, unchanging groove",
                "gentle sidechain pumping",
                "dry rimshot groove",
                "shuffled hats, hypnotic",
                "tight percussion, minimal changes",
            ],
            "bpm": ["118", "120", "121", "122", "124", "125", "126", "128", "130",
                    "132", "134"],
        },
    ),
    # ------------------------------------------------------- hypnotic minimal
    Family(
        "hypnotic_minimal", 2.0, WITH_BPM,
        {
            "style": [
                "Minimal techno for deep focus",
                "Dub techno",
                "Microhouse",
                "Lo-fi house",
                "Detroit-influenced electro",
                "Phase-shifting minimal electronic music",
                "Self-generating modular synth patch",
            ],
            "lead": [
                "tiny repeating marimba-like pattern",
                "deep chord stab drenched in reverb",
                "understated bassline carrying the tune",
                "clean digital sequence, polyrhythmic",
                "warm detuned chords",
                "single-note pulse, slowly filtered",
                "crisp drum machine and a restrained melody",
                "muffled organ chords",
            ],
            "texture": [
                "vinyl crackle and tape hiss",
                "tape delay smearing everything",
                "clicky micro-percussion",
                "subterranean sub bass",
                "dry and close, almost no reverb",
            ],
            "motion": [
                "hypnotic loop, barely changing",
                "clicky percussion, relaxed swing",
                "soft kick pulse, steady",
                "understated shuffled groove",
                "steady pulse with slow filter movement",
                "repetitive and unobtrusive",
            ],
            "bpm": ["112", "114", "116", "118", "120", "121", "122", "124", "125",
                    "126", "128"],
        },
    ),
    # ---------------------------------------------- post-classical + synths
    Family(
        "neo_classical", 1.5, WITH_BPM,
        {
            "style": [
                "Neo-classical electronic instrumental",
                "Post-classical piano with electronics",
                "Modern classical crossover, piano and synth",
                "Chamber electronic instrumental",
                "Nordic neo-classical instrumental",
                "Minimalist classical instrumental, repeating cells",
                "Cinematic neo-classical instrumental",
            ],
            "lead": [
                "arpeggiated felt piano",
                "solo violin over a synth bed",
                "cello and piano in dialogue",
                "pizzicato string sequence",
                "music-box celeste melody",
                "string quartet playing a repeating cell",
                "prepared piano figure",
                "glassy sustained tones",
            ],
            "texture": [
                "soft synth pad underneath",
                "close-miked piano, hammers audible",
                "warm tape saturation",
                "wide reverberant hall",
                "low string drone",
            ],
            "motion": [
                "quiet steady pulse",
                "soft ticking percussion",
                "no drums, momentum from the arpeggio",
                "gentle electronic kick far back in the mix",
                "hypnotic repeating figure",
                "slow rocking motion",
            ],
            "bpm": ["76", "80", "84", "88", "92", "96", "100", "104", "108", "112",
                    "116"],
        },
    ),
    # ------------------------------------------------------- rolling breakbeat
    Family(
        "drum_and_bass", 1.0, WITH_BPM,
        {
            "style": [
                "Liquid drum and bass for focus",
                "Atmospheric drum and bass",
                "Cinematic drum and bass with orchestral elements",
                "Halftime drum and bass, spacious",
                "Deep rolling drum and bass",
                "Jazzy liquid drum and bass",
                "Hypnotic minimal drum and bass",
            ],
            "lead": [
                "lush jazzy chords",
                "warm melodic pads",
                "sampled string phrase, chopped",
                "soft Rhodes chords",
                "distant piano melody",
                "solo violin over the break",
                "muted bell melody",
                "cello line under the drums",
            ],
            "texture": [
                "deep sub bass underneath",
                "cavernous pads",
                "dusty tape-saturated backdrop",
                "wide reverberant space",
                "warm analog bass",
            ],
            "motion": [
                "rolling breakbeat, steady",
                "restrained amen break",
                "crisp two-step drums",
                "propulsive but unobtrusive drums",
                "tight rolling percussion, no drops",
                "steady break, minimal changes",
            ],
            "bpm": ["165", "167", "168", "169", "170", "171", "172", "173", "174",
                    "175", "176"],
        },
    ),
    # ---------------------------------------------------- downtempo / IDM
    Family(
        "downtempo", 1.0, WITH_BPM,
        {
            "style": [
                "Downtempo breakbeat",
                "Glitchy IDM for concentration",
                "Braindance electronica",
                "Downtempo electronica",
                "Chillwave study music",
                "Focused trip-hop instrumental",
                "Ambient breakbeat",
            ],
            "lead": [
                "mellow Rhodes chords",
                "playful melodic synths",
                "hazy detuned synth chords",
                "warm melodic pads",
                "distant piano melody",
                "muted bell melody",
                "soft plucked guitar figure",
                "chopped string sample",
            ],
            "texture": [
                "warm tape saturation",
                "dusty vinyl crackle",
                "deep sub bass underneath",
                "wide atmospheric pads",
                "granular textures drifting",
            ],
            "motion": [
                "dusty chopped drums, relaxed",
                "intricate but soft percussion",
                "crisp broken beat, gentle",
                "slow steady groove",
                "soft head-nod pulse",
                "unhurried shuffled drums",
            ],
            "bpm": ["85", "88", "90", "92", "95", "98", "100", "104", "108", "112",
                    "115"],
        },
    ),
    # ------------------------------------------------------------- ambient
    Family(
        "ambient", 1.0, NO_BPM,
        {
            "style": [
                "Ambient electronic focus music",
                "Generative ambient music",
                "Ambient dub",
                "Kosmische synthesizer music",
                "Berlin school sequencer music",
                "Ambient techno, distant and slow",
                "Cinematic ambient score",
            ],
            "lead": [
                "warm evolving synth pads",
                "overlapping sine tones",
                "gentle bell tones",
                "distant bowed strings",
                "slow analog sequence",
                "deep echoing chords",
                "sustained cello drone",
                "soft glassy arpeggio",
            ],
            "texture": [
                "granular textures drifting",
                "oceanic reverb",
                "tape hiss and wow",
                "wide stereo field",
                "low drone underneath",
            ],
            "motion": [
                "no drums, slow harmonic drift",
                "soft kick pulse, very far back",
                "slow steady pulse",
                "meditative and static",
                "barely-there percussion",
                "gradually evolving, never arriving",
            ],
        },
    ),
]

NEGATIVE_PROMPT = (
    "vocals, singing, choir, vocal chops, spoken word, speech, lyrics, "
    "harsh distortion, sudden loud transitions, dissonance, applause, "
    "crowd noise, low quality"
)


def capacity() -> int:
    """Longest playlist the bank can fill without repeating a prompt.

    Not the sum of the families' combination counts: tracks are handed out by
    weight, so the playlist is capped by whichever family exhausts its
    combinations first -- normally the lightest-weighted one.
    """
    total = sum(f.weight for f in FAMILIES)
    return min(int(f.capacity() * total / f.weight) for f in FAMILIES)


def _allocate(n: int) -> dict:
    """Split n tracks across families by weight, largest remainder first."""
    total = sum(f.weight for f in FAMILIES)
    exact = {f.name: n * f.weight / total for f in FAMILIES}
    counts = {name: int(x) for name, x in exact.items()}

    order = sorted(FAMILIES, key=lambda f: exact[f.name] - counts[f.name], reverse=True)
    for f in order[: n - sum(counts.values())]:
        counts[f.name] += 1
    return counts


def _draw(family: Family, n: int, rng) -> list:
    """n distinct prompts from `family`, each differing from the last in every slot.

    Shuffling the pools and then stepping every one of them at once means slot k
    repeats on a cycle of len(pool_k), and the combination only repeats at the
    lcm of those lengths -- so the prompts stay unique, and adjacent draws never
    share so much as a bpm.
    """
    if n > family.capacity():
        raise ValueError(f"{family.name}: asked for {n} prompts, capacity is "
                         f"{family.capacity()}")

    pools = {slot: list(values) for slot, values in family.slots.items()}
    for values in pools.values():
        rng.shuffle(values)

    return [
        family.template.format(
            **{slot: values[i % len(values)] for slot, values in pools.items()}
        )
        for i in range(n)
    ]


def build_prompts(n: int, rng) -> tuple:
    """Return (prompts, per-family counts): n distinct prompts, families interleaved.

    Each family's tracks are spread evenly across the playlist by sorting on a
    stratified key, so you never get four ambient tracks back to back.
    """
    if n > capacity():
        raise ValueError(f"asked for {n} prompts, bank capacity is {capacity()}")

    counts = _allocate(n)
    keyed = []
    for family in FAMILIES:
        drawn = _draw(family, counts[family.name], rng)
        for i, prompt in enumerate(drawn):
            keyed.append(((i + rng.random()) / len(drawn), prompt))

    keyed.sort(key=lambda pair: pair[0])
    return [prompt for _, prompt in keyed], counts
