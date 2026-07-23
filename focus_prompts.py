"""Prompt bank for the focus playlist.

Eight hours at 120s per track is 240 tracks, and a hand-written bank that long
either repeats or drifts into one texture. So prompts are built combinatorially:
each family is a sub-style with its own slot pools, and a prompt is one value
drawn from each pool. Pool sizes are pairwise coprime-ish, so cycling them
odometer-style walks a long path before any combination comes back around --
lcm(5,6,7,8,11) = 9240 prompts per family, which is more than we will ever ask
for. The allocation/drawing/interleaving machinery lives in prompt_bank.py.

Families are tuned for aggressive, locked-in focus: relentless, hard-driving,
full-intensity material that pins your attention and does not let it wander.
Hard electronic and string-led crossover carry the playlist, and every family
hammers forward -- nothing mellow, ambient or downtempo to drift out on. Two
constraints hold the aggression to focus music rather than chaos: nothing here
has vocals (no slot value mentions voices, choirs or lyrics -- the model will
happily sing if invited), and nothing drops out. Energy stays pinned; there are
no build-ups that release and no breakdowns to break the trance.
"""

from prompt_bank import Family
from prompt_bank import build_prompts as _build_prompts
from prompt_bank import capacity as _capacity


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
                "Relentless action film score, electronic and orchestral",
                "Driving hybrid orchestral-electronic score, full intensity",
                "Widescreen chase-sequence score",
                "Trailer music at full tilt, no let-up",
                "High-stakes heist score, precise and pounding",
                "Sci-fi battle score, cold and merciless",
                "Pursuit montage score, hard and propulsive",
            ],
            "lead": [
                "hammering string ostinato",
                "urgent piano motif pounding in octaves",
                "solo violin sawing a fierce line",
                "brass stabs cutting through",
                "razor-edged synth lead, driving",
                "cello line grinding in fast steps",
                "aggressive plucked sequence, machine-tight",
                "bowed metallic melody, tense",
            ],
            "texture": [
                "distorted analog pads pushing underneath",
                "granular string textures, agitated",
                "gritty tape-saturated strings",
                "dense reverberant wall of sound",
                "menacing low drone driving the harmony",
            ],
            "motion": [
                "pounding taiko floor toms",
                "insistent hammering low pulse",
                "hard driving kick, relentless",
                "militaristic snare marking time",
                "staccato strings stabbing the beat",
                "relentless sixteenth-note percussion",
            ],
            "bpm": ["120", "124", "128", "130", "132", "134", "136", "138", "140",
                    "144", "150"],
        },
    ),
    # ------------------------------------------------- electric violin over EDM
    Family(
        "violin_electronic", 3.0, WITH_BPM,
        {
            "style": [
                "Aggressive electronic violin crossover instrumental",
                "High-energy dance instrumental built around a shredding violin",
                "Celtic-tinged electronic instrumental, fierce violin lead",
                "Epic cinematic dance instrumental, violin lead at full power",
                "Folk-electronic crossover, fiddle driving hard over programmed drums",
                "Epic violin electronica, relentless",
                "Middle-Eastern-inflected electronic banger, violin lead",
            ],
            "lead": [
                "furious staccato violin riff",
                "soaring violin melody pushed to the limit",
                "double-stopped violin hook, aggressive",
                "machine-gun pizzicato violin ostinato",
                "violin trading fierce phrases with a distorted synth",
                "blistering sixteenth-note violin line",
                "violin melody doubled in screaming octaves",
                "keening violin countermelody cutting over a driving string section",
            ],
            "texture": [
                "layered string section pushing underneath",
                "huge distorted supersaw chords",
                "gritty analog pads and hard plucks",
                "orchestral strings and heavy synth bass together",
                "aggressive bell stabs",
            ],
            "motion": [
                "pounding four-on-the-floor kick",
                "hard-hitting programmed drums, no drops",
                "driving syncopated beat, relentless",
                "punchy kick and clap, full throttle",
                "rolling hand drums at full intensity",
                "steady relentless drive, no breakdowns",
            ],
            "bpm": ["124", "128", "130", "132", "134", "136", "138", "140", "144",
                    "148", "150"],
        },
    ),
    # ------------------------------------------------ layered piano and cellos
    Family(
        "piano_cello", 2.0, WITH_BPM,
        {
            "style": [
                "Percussive piano and cello instrumental, driving hard",
                "Aggressive piano and cello duel, cinematic",
                "Pounding piano and cello arrangement, full intensity",
                "Piano and cello score, urgent and relentless",
                "Rhythmic piano and cello crossover, propulsive",
                "Dark piano and cello instrumental, hammering",
                "Epic piano and cello anthem, driving",
            ],
            "lead": [
                "hammering piano arpeggios",
                "fierce cello melody in the upper register",
                "pounding octave piano melody",
                "cello and piano trading aggressive lines",
                "relentless rolling left-hand piano ostinato",
                "cello double-stops driving the theme hard",
                "urgent piano melody, no let-up",
                "cascading piano figure over a grinding cello pedal",
            ],
            "texture": [
                "stacked cello harmonies pushing underneath",
                "hard pizzicato cellos hammering the pulse",
                "tight close-miked room, aggressive",
                "grinding low cello drone holding the key",
                "heavy synth pad under the acoustic instruments",
            ],
            "motion": [
                "hard cello body percussion driving time",
                "pounding driving eighth notes",
                "relentless cajon pulse",
                "propulsive staccato bass line, aggressive",
                "stomping four-on-the-floor feel",
                "tight relentless groove, no let-up",
            ],
            "bpm": ["108", "112", "116", "120", "124", "128", "130", "132", "136",
                    "140", "144"],
        },
    ),
    # ------------------------------------------------------ driving cello duo
    Family(
        "cello_rock", 2.5, WITH_BPM,
        {
            "style": [
                "Cello metal instrumental, two cellos, full aggression",
                "Dramatic cello duo instrumental, hard rock energy",
                "Cello-driven instrumental with a pounding rock rhythm section",
                "Symphonic metal instrumental led by cellos",
                "Aggressive classical crossover for shredding cellos",
                "Cello ensemble instrumental, driving and savage",
                "Baroque-flavored cello metal instrumental",
            ],
            "lead": [
                "heavily distorted cello riff",
                "furious sawing cello ostinato",
                "soaring cello melody screaming over the riff",
                "two cellos trading savage lines",
                "cello playing a menacing bass riff",
                "chugging palm-muted cello chords",
                "cello melody in the high register, ferocious",
                "cellos in unison octaves, full force",
            ],
            "texture": [
                "string section slamming behind",
                "gritty high-gain amp tone on the low cello",
                "huge room reverb",
                "layered cello harmonies, dense",
                "heavy synth bass reinforcing the low end",
            ],
            "motion": [
                "hard-hitting rock drums, relentless",
                "pounding kick and snare, driving",
                "relentless eighth-note pulse",
                "tight aggressive groove, no tempo changes",
                "punchy drums, sustained full intensity",
                "propulsive stomping half-time groove",
            ],
            "bpm": ["112", "116", "120", "124", "128", "132", "136", "140", "144",
                    "150", "160"],
        },
    ),
    # --------------------------------------------- steady electronic momentum
    Family(
        "driving", 3.0, WITH_BPM,
        {
            "style": [
                "Peak-time driving techno for locked-in focus",
                "Relentless progressive house for concentration",
                "Hard driving electronic coding music",
                "Pounding rolling techno for deep work",
                "High-energy melodic house for concentration",
                "Trance-influenced focus banger, full drive",
                "Cinematic peak-time techno",
            ],
            "lead": [
                "hypnotic arpeggio cycling hard",
                "aggressive plucked synth motif",
                "hard supersaw chord stabs",
                "driving string ostinato over the beat",
                "piercing bell sequence",
                "screaming acid line",
                "filter cranking open on a saw lead",
                "detuned pluck sequence, relentless",
            ],
            "texture": [
                "wide aggressive evolving pads",
                "grinding analog bass underneath",
                "hard reverb-slammed chord stabs",
                "agitated granular textures in the background",
                "pounding sub bass and little else",
            ],
            "motion": [
                "pounding four-on-the-floor kick, no drops",
                "relentless rolling bassline, unchanging",
                "hard rimshot groove",
                "driving hats, hypnotic and tight",
                "relentless percussion, minimal changes",
                "stomping kick, full throttle",
            ],
            "bpm": ["126", "128", "130", "132", "134", "136", "138", "140", "142",
                    "144", "150"],
        },
    ),
    # ------------------------------------------------------- hard acid drive
    Family(
        "acid_drive", 2.0, WITH_BPM,
        {
            "style": [
                "Driving acid techno for hyperfocus",
                "Hard rolling techno, relentless",
                "Pumping electro, aggressive and tight",
                "Acid-fueled focus banger",
                "Detroit-influenced hard electro",
                "Relentless driving modular synth workout",
                "Peak-time acid trance instrumental",
            ],
            "lead": [
                "screaming 303 acid line",
                "aggressive chord stab drenched in reverb",
                "grinding bassline carrying the tune",
                "hard digital sequence, polyrhythmic and tight",
                "detuned chords pushed hard",
                "single-note pulse, filter slamming",
                "crisp drum machine and a relentless riff",
                "distorted organ stabs",
            ],
            "texture": [
                "gritty distortion and tape saturation",
                "hard tape delay driving everything",
                "clicky aggressive micro-percussion",
                "pounding subterranean sub bass",
                "dry and hard, in-your-face",
            ],
            "motion": [
                "relentless loop, hammering forward",
                "hard clicky percussion, tight",
                "pounding kick pulse, driving",
                "aggressive shuffled groove",
                "steady pounding pulse with slamming filter movement",
                "relentless and unbroken",
            ],
            "bpm": ["128", "130", "132", "134", "136", "138", "140", "142", "144",
                    "145", "150"],
        },
    ),
    # ------------------------------------------------------- rolling breakbeat
    Family(
        "drum_and_bass", 1.5, WITH_BPM,
        {
            "style": [
                "Rolling neurofunk drum and bass for focus",
                "High-energy drum and bass, relentless",
                "Cinematic drum and bass with pounding orchestral elements",
                "Driving jump-up-tinged drum and bass, tight",
                "Deep rolling drum and bass at full drive",
                "Aggressive liquid drum and bass",
                "Relentless minimal drum and bass",
            ],
            "lead": [
                "hard-edged jazzy chords",
                "driving melodic pads",
                "chopped string phrase, aggressive",
                "hard Rhodes stabs",
                "urgent piano melody",
                "fierce solo violin over the break",
                "piercing bell melody",
                "grinding cello line under the drums",
            ],
            "texture": [
                "pounding sub bass underneath",
                "dense cavernous pads",
                "gritty tape-saturated backdrop",
                "wide aggressive reverberant space",
                "grinding analog bass",
            ],
            "motion": [
                "rolling breakbeat, relentless",
                "hard-driving amen break",
                "crisp aggressive two-step drums",
                "propulsive pounding drums",
                "tight rolling percussion, no drops",
                "relentless break, full drive",
            ],
            "bpm": ["168", "170", "172", "173", "174", "175", "176", "177", "178",
                    "180", "182"],
        },
    ),
]

NEGATIVE_PROMPT = (
    "vocals, singing, choir, vocal chops, spoken word, speech, lyrics, "
    "ambient, downtempo, mellow, sparse, gentle, meandering, relaxing, "
    "beatless, new age, big drops, breakdowns, sudden silence, "
    "applause, crowd noise, low quality"
)


def capacity() -> int:
    """Longest focus playlist the bank can fill without repeating a prompt."""
    return _capacity(FAMILIES)


def build_prompts(n: int, rng) -> tuple:
    """Return (prompts, per-family counts): n distinct prompts, families interleaved."""
    return _build_prompts(n, FAMILIES, rng)
