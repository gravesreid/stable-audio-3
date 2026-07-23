"""Prompt bank for the lifting playlist.

Same combinatorial engine as the focus bank (see prompt_bank.py): each family is
a sub-style with its own slot pools, and a prompt draws one value from each pool,
so a short hand-written family expands into thousands of distinct prompts and a
long session never repeats.

Where the focus bank spreads across cinematic and string crossover, this one has
a single job: mega-aggressive, four-on-the-floor material to lift heavy shit for
reps. Every family is a pounding steady kick on every beat -- hard techno,
hardstyle, hardgroove, hard dance, industrial, acid, gabber, and one epic
orchestral-rave nod to the string crossover taste -- so the pulse never drops out
from under a set. Three constraints keep it lift-able rather than chaotic: no
vocals (no slot value mentions voices, choirs or lyrics -- the model will happily
sing if invited), no breakbeats or half-time (the steady 4/4 is what you rep to),
and nothing drops out. There is no warmup and no cooldown; it hammers from track
one.
"""

from prompt_bank import Family
from prompt_bank import build_prompts as _build_prompts
from prompt_bank import capacity as _capacity


# Every family uses this shape, so prompts read consistently to the model:
#   <style>, <lead>, <texture>, <motion>, no vocals, <bpm> BPM
# The <motion> pool is always a pounding four-on-the-floor kick -- that is the
# whole point of this bank, so it is the one slot that never breaks character.
TEMPLATE = "{style}, {lead}, {texture}, {motion}, no vocals, {bpm} BPM"

FAMILIES = [
    # ----------------------------------------------------------- hard techno
    Family(
        "hard_techno", 3.0, TEMPLATE,
        {
            "style": [
                "Peak-time hard techno, brutal and relentless",
                "Rumbling warehouse hard techno, full intensity",
                "Pounding hard techno for lifting heavy",
                "Dark industrial-tinged hard techno, merciless",
                "Hypnotic hard techno, hammering forward",
                "Uptempo hard techno banger, no let-up",
                "Schranz-influenced hard techno, savage",
            ],
            "lead": [
                "distorted siren stab riff",
                "screaming detuned lead cutting through",
                "hard reverb-slammed chord stabs",
                "menacing hoover stab sequence",
                "grinding acid line, relentless",
                "piercing metallic stab loop",
                "aggressive rolling synth riff",
                "dark tunnel-reverb rave stab",
            ],
            "texture": [
                "rumbling distorted sub bass underneath",
                "gritty tape-saturated wall of noise",
                "menacing low drone driving the harmony",
                "hard-panned industrial percussion clatter",
                "cavernous warehouse reverb",
            ],
            "motion": [
                "pounding four-on-the-floor kick, rumbling and distorted",
                "relentless hard kick on every beat",
                "hammering distorted kick, unbroken",
                "driving four-on-the-floor, offbeat rumble bass",
                "stomping kick, full throttle, no drops",
                "tight relentless 4/4 pulse, sustained intensity",
            ],
            "bpm": ["145", "146", "148", "150", "152", "154", "155", "156", "158",
                    "160"],
        },
    ),
    # -------------------------------------------------------------- hardstyle
    Family(
        "hardstyle", 2.5, TEMPLATE,
        {
            "style": [
                "Raw hardstyle instrumental, full power",
                "Euphoric hardstyle, enormous energy",
                "Rawstyle instrumental, brutal and screeching",
                "Xtra raw hardstyle, punishing",
                "Melodic hardstyle anthem, driving hard",
                "Uptempo hardstyle, relentless",
                "Dark hardstyle, menacing and huge",
            ],
            "lead": [
                "screeching distorted rawstyle lead",
                "euphoric supersaw melody, soaring",
                "detuned screech lead, aggressive",
                "pitched-up distorted synth hook",
                "menacing reverse-bass melody",
                "huge anthem lead, full force",
                "gritted screech riff, relentless",
                "hard pitched stab sequence",
            ],
            "texture": [
                "wide distorted supersaw chords",
                "gritty overdriven midrange, dense",
                "huge reverberant hardstyle atmosphere",
                "screaming FM stabs layered underneath",
                "dark orchestral hit reinforcing the drop",
            ],
            "motion": [
                "colossal distorted hardstyle kick on every beat",
                "pounding reverse-bass four-on-the-floor",
                "hammering distorted kick, relentless",
                "stomping 4/4 kick, full throttle",
                "hard-hitting kick and screech, no drops",
                "relentless pounding kick drum, unbroken",
            ],
            "bpm": ["148", "150", "151", "152", "153", "154", "155", "156", "158"],
        },
    ),
    # ------------------------------------------------------- rolling hardgroove
    Family(
        "hardgroove", 2.5, TEMPLATE,
        {
            "style": [
                "Rolling hard groove techno, tribal and relentless",
                "Peak-time hardgroove, insistent and driving",
                "Percussive hard groove techno, full tilt",
                "Tribal hard techno groove, hammering",
                "Loopy rolling hardgroove, hypnotic and hard",
                "Jeff Mills-style driving techno, relentless",
                "Pumping hard groove, no let-up",
            ],
            "lead": [
                "hypnotic rolling stab loop",
                "aggressive tribal tom melody",
                "hard percussive synth riff, driving",
                "menacing bleep sequence, relentless",
                "filtered stab hook cycling hard",
                "grinding bass riff carrying the groove",
                "sharp clave-and-stab pattern",
                "dark rolling arp, hammering",
            ],
            "texture": [
                "dense layered tribal percussion",
                "rumbling rolling sub bass underneath",
                "gritty analog hats and shakers driving",
                "cavernous reverb on the toms",
                "grinding low drone holding the key",
            ],
            "motion": [
                "pounding four-on-the-floor kick, rolling toms between",
                "relentless 4/4 kick with tribal percussion",
                "driving kick on every beat, unbroken groove",
                "hard rolling kick and toms, full intensity",
                "stomping four-on-the-floor, hypnotic and tight",
                "tight relentless kick pulse, no drops",
            ],
            "bpm": ["135", "136", "138", "140", "141", "142", "143", "144", "145",
                    "146"],
        },
    ),
    # ---------------------------------------------------------- hard dance stomp
    Family(
        "rave_stomp", 2.0, TEMPLATE,
        {
            "style": [
                "Hard house stomper, pounding and relentless",
                "UK hardcore-tinged hard dance, full energy",
                "Donk banger, bouncing and aggressive",
                "Scouse house stomp, relentless drive",
                "Hard bounce instrumental, huge and driving",
                "Old-school hard rave stomper, full tilt",
                "Pumping hard dance, no let-up",
            ],
            "lead": [
                "classic hoover stab riff, aggressive",
                "bouncing donk bassline hook",
                "piercing rave stab sequence",
                "detuned organ stab, driving",
                "hard pitched-up melodic hook",
                "screaming saw lead, relentless",
                "offbeat bassline stab, pumping",
                "euphoric riff pushed to the limit",
            ],
            "texture": [
                "pounding offbeat bassline underneath",
                "gritty distorted midrange stabs",
                "huge reverberant rave atmosphere",
                "hard-slammed reverb on the stabs",
                "thick compressed wall of synths",
            ],
            "motion": [
                "stomping four-on-the-floor kick, offbeat bass",
                "pounding 4/4 kick, bouncing donk between beats",
                "relentless hard kick on every beat",
                "driving four-on-the-floor, full throttle",
                "hammering kick and offbeat bass, unbroken",
                "tight pounding four-on-the-floor, no breakdowns",
            ],
            "bpm": ["138", "140", "142", "144", "145", "146", "148", "150", "152"],
        },
    ),
    # ------------------------------------------------------ industrial techno
    Family(
        "industrial", 2.0, TEMPLATE,
        {
            "style": [
                "Industrial techno, metallic and merciless",
                "Distorted industrial techno, pounding",
                "Berlin-style industrial techno, cold and hard",
                "Rhythmic noise-tinged techno, brutal",
                "Dark industrial techno, hammering forward",
                "Warehouse industrial techno, full intensity",
                "Mechanical industrial techno, relentless",
            ],
            "lead": [
                "grinding distorted stab loop",
                "metallic clanging riff, rhythmic",
                "menacing feedback drone melody",
                "hard-edged noise stab sequence",
                "cold detuned lead cutting through",
                "corroded acid line, relentless",
                "machine-tight bleep pattern",
                "distorted alarm-stab hook",
            ],
            "texture": [
                "clattering metallic percussion, hard-panned",
                "rumbling distorted sub bass underneath",
                "gritty overdriven noise floor",
                "cavernous concrete reverb",
                "grinding feedback drone holding the key",
            ],
            "motion": [
                "pounding four-on-the-floor kick, distorted and metallic",
                "relentless hard kick with clanging percussion",
                "hammering 4/4 kick, mechanical and unbroken",
                "driving distorted kick on every beat",
                "stomping industrial kick, full intensity",
                "tight relentless four-on-the-floor, no drops",
            ],
            "bpm": ["135", "136", "138", "140", "142", "143", "144", "145", "146",
                    "148"],
        },
    ),
    # ------------------------------------------------------------- hard acid
    Family(
        "acid", 2.0, TEMPLATE,
        {
            "style": [
                "Hard acid techno, screaming and relentless",
                "Pumping acid banger, full throttle",
                "Peak-time acid techno, hypnotic brutality",
                "Distorted acid workout, driving hard",
                "Raw acid techno, menacing and tight",
                "Rolling acid techno, relentless",
                "Overdriven acid stomper, no let-up",
            ],
            "lead": [
                "screaming 303 acid line, filter slamming",
                "squelching acid riff, relentless",
                "distorted 303 sequence cranking open",
                "twin acid lines writhing together",
                "hard resonant acid hook",
                "detuned acid stab pattern",
                "grinding acid bassline carrying the tune",
                "piercing acid lead, full drive",
            ],
            "texture": [
                "gritty distortion and tape saturation",
                "rumbling sub bass underneath",
                "hard tape delay driving everything",
                "clicky aggressive micro-percussion",
                "dry and hard, in-your-face",
            ],
            "motion": [
                "pounding four-on-the-floor kick, acid rolling over it",
                "relentless hard kick on every beat",
                "driving 4/4 kick, filter slamming the acid",
                "hammering kick, unbroken and tight",
                "stomping four-on-the-floor, full throttle",
                "tight relentless four-on-the-floor, no breakdowns",
            ],
            "bpm": ["138", "140", "142", "143", "144", "145", "146", "148", "150",
                    "152"],
        },
    ),
    # --------------------------------------------------------- gabber / hardcore
    Family(
        "gabber", 1.5, TEMPLATE,
        {
            "style": [
                "Gabber instrumental, overdriven kick barrage",
                "Rotterdam hardcore, brutal and raw",
                "Frenchcore instrumental, relentless kick drive",
                "Uptempo hardcore, punishing intensity",
                "Mainstream hardcore, huge and driving",
                "Industrial hardcore, distorted and merciless",
                "Terror-tinged gabber, full aggression",
            ],
            "lead": [
                "distorted hardcore stab riff",
                "screaming pitched hoover lead",
                "menacing gabber synth hook",
                "hard detuned stab sequence",
                "pitched-up screech lead, relentless",
                "grinding distorted riff, full force",
                "aggressive rave stab loop",
                "piercing alarm lead cutting through",
            ],
            "texture": [
                "wall of distorted kick harmonics",
                "gritty overdriven noise, dense",
                "huge reverberant hardcore atmosphere",
                "corroded distortion on everything",
                "rumbling distorted low end",
            ],
            "motion": [
                "overdriven four-on-the-floor gabber kick, distorted to a roar",
                "relentless distorted kick barrage on every beat",
                "hammering hardcore kick, unbroken",
                "pounding 4/4 gabber kick, full throttle",
                "stomping distorted kick, punishing and tight",
                "relentless pounding kick, no let-up",
            ],
            "bpm": ["155", "160", "165", "170", "175", "180", "185", "190"],
        },
    ),
    # ------------------------------------------------ epic orchestral rave nod
    Family(
        "epic_rave", 1.5, TEMPLATE,
        {
            "style": [
                "Epic cinematic rave, orchestral and pounding",
                "Hybrid trailer-rave, huge and driving",
                "Cinematic hard trance, aggressive and soaring",
                "Orchestral techno banger, full intensity",
                "Epic hardstyle-orchestral crossover, relentless",
                "Battle-rave instrumental, brass and distorted synths",
                "Symphonic hard dance, enormous and driving",
            ],
            "lead": [
                "soaring string ostinato over the beat",
                "aggressive brass stab riff",
                "epic supersaw lead doubling the strings",
                "furious solo violin line, driving",
                "huge orchestral hit hook",
                "screaming saw lead trading with brass",
                "pounding octave piano motif",
                "distorted synth lead over hammering strings",
            ],
            "texture": [
                "layered orchestral strings pushing underneath",
                "huge distorted supersaw chords",
                "cinematic brass swells reinforcing the drop",
                "grinding synth bass under the orchestra",
                "wide reverberant epic atmosphere",
            ],
            "motion": [
                "pounding four-on-the-floor kick under the orchestra",
                "relentless hard kick with taiko hits",
                "driving 4/4 kick, orchestral stabs on the beat",
                "hammering kick, full intensity, no drops",
                "stomping four-on-the-floor, epic and relentless",
                "tight pounding four-on-the-floor, sustained power",
            ],
            "bpm": ["138", "140", "142", "144", "145", "146", "148", "150", "152",
                    "155"],
        },
    ),
]

# Ballads and ambience kill a set; so does anything that breaks the steady pulse
# you rep to. Reject vocals, low-energy textures, and the syncopated/half-time
# grooves (breakbeat, drum and bass) that pull against strict four-on-the-floor.
NEGATIVE_PROMPT = (
    "vocals, singing, choir, vocal chops, spoken word, lyrics, "
    "ambient, downtempo, mellow, gentle, sparse, acoustic, relaxing, "
    "slow tempo, low energy, breakbeat, drum and bass, half-time, "
    "big drops, sudden silence, low quality"
)


def capacity() -> int:
    """Longest lifting playlist the bank can fill without repeating a prompt."""
    return _capacity(FAMILIES)


def build_prompts(n: int, rng) -> tuple:
    """Return (prompts, per-family counts): n distinct prompts, families interleaved."""
    return _build_prompts(n, FAMILIES, rng)
