"""Tests for the mellow playlist prompt bank (mellow_prompts.py).

The bank makes three promises, and all three are easy to break by hand-editing a
pool:

1. Thousands of distinct prompts, so a long session never repeats.
2. Every prompt is shaped like the AudioSparx metadata the model was trained on
   -- the labelled fields, in the order the prompting guides give.
3. Nothing loud, nothing eventful, and *no negations* -- "no drums" in a positive
   prompt asks the text encoder to represent absence, which it cannot do, and
   naming drums at all makes drums likelier. Absence is expressed with
   VocalType/TrackType/Tempo instead, and that is what these tests enforce.

Pure data checks -- no model, no torch, no weights.
"""

import random
import re
import string
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mellow_prompts import (  # noqa: E402
    FAMILIES,
    NEGATIVE_PROMPT,
    build_prompts,
    capacity,
)

# A long unwinding session at the 120s per-track ceiling of the small-music
# model: four hours. The bank's real capacity is many times this; the point of
# the test is that nothing repeats however long you leave it on.
LONG_SESSION = 120

# Anything that might invite a vocal take. Matched on word boundaries so
# "lyrical" and "rising" don't trip "lyrics" and "sing".
VOCAL_WORDS = [
    "vocal", "vocals", "voice", "voices", "choir", "choral", "sing", "singing",
    "sung", "lyrics", "lyric", "chant", "chanting", "spoken", "speech",
    "acapella", "vox", "soprano", "tenor", "baritone", "humming",
]
VOCAL_RE = re.compile(rf"\b({'|'.join(VOCAL_WORDS)})\b", re.IGNORECASE)

# The words that make music stop being background. Two kinds: loudness/attack
# ("pounding", "distorted"), and eventfulness ("build-up", "drop") -- a playlist
# you leave on for hours should never make you look up.
LOUD_WORDS = [
    "aggressive", "pounding", "relentless", "hammering", "driving", "distorted",
    "screaming", "piercing", "brutal", "stomping", "punchy", "banger", "hard",
    "harsh", "loud", "shrill", "menacing", "furious", "savage", "shredding",
    "pumping", "throttle", "frantic", "urgent", "fast", "uptempo", "energetic",
    "intense", "slam", "slamming", "stab", "stabs", "crash", "drop", "drops",
    "sudden", "abrupt",
]
LOUD_RE = re.compile(rf"\b({'|'.join(LOUD_WORDS)})\b", re.IGNORECASE)

# Negation in a positive prompt is the bug this bank was rewritten to remove.
# "beatless" and "wireless" are fine -- this is about "no X" / "without X".
NEGATION_RE = re.compile(r"\b(no|not|none|never|without|avoid|absent|lacking|"
                         r"free of|minus)\b", re.IGNORECASE)

# Nothing in this bank should be faster than a resting heart rate plus a bit.
BPM_CEILING = 115

# Words that ask the model to stop playing. Measured, not theoretical: with "long
# pauses" and "sparse" in the detail pools, the solo-instrument families produced
# 120s tracks that were 30-60% near-silence, which reads as the playlist having
# died rather than as restful space.
EMPTY_WORDS = ["pause", "pauses", "silence", "silent", "sparse", "empty", "gap",
               "gaps", "stops", "stopping"]
EMPTY_RE = re.compile(rf"\b({'|'.join(EMPTY_WORDS)})\b", re.IGNORECASE)

# The field order both guides prescribe, as it appears in a finished prompt.
FIELD_ORDER = ["TrackType:", "VocalType:", "Genre:", "Subgenre:", "Instruments:",
               "Moods:"]

# Specific instruments a Subgenre value might name. If a family's subgenre commits
# to one of these, every one of its instruments values has to agree, or the two
# fields contradict each other -- "Subgenre: Music Box Miniature | Instruments:
# nylon-string classical guitar" is exactly the incoherence the guides warn about.
# Deliberately specific: category words like "strings" would flag "String Adagio"
# against "cello, viola, violin", which is not a contradiction at all.
INSTRUMENT_WORDS = ["piano", "guitar", "harp", "music box", "cello", "violin",
                    "kalimba", "dulcimer", "organ", "marimba", "vibraphone"]


def _slot_values():
    for family in FAMILIES:
        for slot, values in family.slots.items():
            for value in values:
                yield family.name, slot, value


# ---------------------------------------------------------------------------
# Bank structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.name)
def test_template_slots_match_pools(family):
    """Every {slot} in the template has a pool, and every pool is used."""
    referenced = {
        name for _, name, _, _ in string.Formatter().parse(family.template) if name
    }
    assert referenced == set(family.slots), (
        f"{family.name}: template uses {referenced}, pools define {set(family.slots)}"
    )


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.name)
def test_pool_values_are_distinct(family):
    """A repeated value silently shrinks the family's real capacity."""
    for slot, values in family.slots.items():
        dupes = {v for v in values if values.count(v) > 1}
        assert not dupes, f"{family.name}.{slot} repeats: {dupes}"


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.name)
def test_pools_are_non_trivial(family):
    """Every pool needs 2+ values, or adjacent prompts stop differing in that slot."""
    for slot, values in family.slots.items():
        assert len(values) >= 2, f"{family.name}.{slot} has {len(values)} value(s)"


def test_family_names_are_unique():
    names = [f.name for f in FAMILIES]
    assert len(set(names)) == len(names)


def test_capacity_covers_a_long_session():
    assert capacity() >= LONG_SESSION


# ---------------------------------------------------------------------------
# Prompt shape: the metadata structure the guides prescribe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.name)
def test_prompts_lead_with_tracktype_and_vocaltype(family):
    """The two tags the guide credits for coherence, in the position it shows."""
    assert family.template.startswith("TrackType: ")
    assert "VocalType: Instrumental" in family.template


def test_every_prompt_declares_itself_instrumental():
    """This replaces the old "no vocals" negation, and does the same job."""
    prompts, _ = build_prompts(LONG_SESSION, random.Random(0))
    assert all("VocalType: Instrumental" in p for p in prompts)


def test_fields_appear_in_the_guides_order():
    """Both guides say the order matters, so pin it."""
    prompts, _ = build_prompts(LONG_SESSION, random.Random(0))
    for prompt in prompts:
        positions = [prompt.index(field) for field in FIELD_ORDER]
        assert positions == sorted(positions), f"fields out of order: {prompt}"


def test_every_prompt_states_a_tempo():
    """Either an explicit BPM or Tempo: Slow -- the positive form of "no beat"."""
    prompts, _ = build_prompts(LONG_SESSION, random.Random(0))
    for prompt in prompts:
        assert "BPM: " in prompt or "Tempo: Slow" in prompt, prompt


def test_a_prompt_ends_with_one_sentence_of_detail():
    """Fields carry the metadata; the prose tail carries technique and room."""
    prompts, _ = build_prompts(LONG_SESSION, random.Random(0))
    for prompt in prompts:
        tail = prompt.split(". ", 1)[1] if ". " in prompt else ""
        assert tail.endswith("."), f"no prose detail sentence: {prompt}"
        assert "|" not in tail, f"detail must come after the fields: {prompt}"


def test_beatless_families_have_no_bpm_slot():
    """A bpm on ambient/drone material is an invitation to add a beat to it."""
    for family in FAMILIES:
        if "bpm" in family.slots:
            assert "{bpm}" in family.template, f"{family.name} has an unused bpm pool"
        else:
            assert "Tempo: Slow" in family.template, family.name


def test_tempos_stay_slow():
    for name, slot, value in _slot_values():
        if slot == "bpm":
            assert int(value) <= BPM_CEILING, f"{name}.bpm: {value} is too fast"


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.name)
def test_subgenre_does_not_contradict_instruments(family):
    """A Subgenre naming an instrument commits every prompt in the family to it."""
    instruments = [v.lower() for v in family.slots["instruments"]]
    for subgenre in family.slots["subgenre"]:
        for word in INSTRUMENT_WORDS:
            # Word-boundaried both ends, so "Organic Downtempo" is not an organ.
            if re.search(rf"\b{word}\b", subgenre, re.IGNORECASE):
                assert all(word in i for i in instruments), (
                    f"{family.name}: 'Subgenre: {subgenre}' names {word!r}, but "
                    f"its instruments pool does not always agree"
                )


# ---------------------------------------------------------------------------
# Generated prompts: distinctness and calm
# ---------------------------------------------------------------------------


def test_long_session_of_prompts_are_all_distinct():
    prompts, _ = build_prompts(LONG_SESSION, random.Random(0))
    assert len(prompts) == LONG_SESSION
    assert len(set(prompts)) == LONG_SESSION


def test_prompts_are_distinct_at_full_capacity():
    """The uniqueness claim should hold at the advertised ceiling, not just at 120."""
    prompts, _ = build_prompts(capacity(), random.Random(0))
    assert len(set(prompts)) == capacity()


@pytest.mark.parametrize("n", [1, 7, 30, 100, LONG_SESSION, 1000])
def test_counts_sum_to_requested_length(n):
    prompts, counts = build_prompts(n, random.Random(n))
    assert len(prompts) == n
    assert sum(counts.values()) == n


def test_no_prompt_contains_a_negation():
    """The whole reason for the rewrite: state what you want, never what you don't."""
    prompts, _ = build_prompts(LONG_SESSION, random.Random(0))
    for prompt in prompts:
        match = NEGATION_RE.search(prompt)
        assert not match, f"{prompt!r} negates with {match.group(0)!r}"


def test_no_slot_value_invites_a_vocal():
    for name, slot, value in _slot_values():
        match = VOCAL_RE.search(value)
        assert not match, f"{name}.{slot}: {value!r} contains {match.group(0)!r}"


# The prose sentence has to say what is being *played*. A detail value that only
# describes the recording -- "Recorded close in a warm wooden room, faint key noise
# and tape hiss" -- leaves the model nothing to play, and it obliges: those prompts
# produced the 37% and 60% near-silent tracks.
PLAYING_WORDS = [
    "melody", "chord", "chords", "arpeggio", "arpeggios", "pattern", "figure",
    "groove", "line", "lines", "phrase", "phrases", "phrasing", "tone", "tones",
    "drone", "harmonics", "pizzicato", "bass", "swells", "overtones", "texture",
    "strings", "cycling", "picked", "hats", "harmony", "drums", "kick", "pulse",
    "beat", "percussion", "note", "notes", "snare", "pad", "pads", "sustained",
]
PLAYING_RE = re.compile(rf"\b({'|'.join(PLAYING_WORDS)})\b", re.IGNORECASE)


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.name)
def test_every_detail_says_what_is_played(family):
    for value in family.slots["detail"]:
        assert PLAYING_RE.search(value), (
            f"{family.name}: detail {value!r} describes the recording but names "
            f"nothing being played"
        )


def test_no_slot_value_asks_the_music_to_stop():
    """Restful is not the same as absent: background music has to keep playing."""
    for name, slot, value in _slot_values():
        match = EMPTY_RE.search(value)
        assert not match, f"{name}.{slot}: {value!r} contains {match.group(0)!r}"


def test_no_slot_value_is_loud_or_eventful():
    """One hard transient is what pulls your ear back off what you were doing."""
    for name, slot, value in _slot_values():
        match = LOUD_RE.search(value)
        assert not match, f"{name}.{slot}: {value!r} contains {match.group(0)!r}"


# Moods carry the calm now that the old prose "motion" slot is gone. Each value is
# a list of AudioSparx-style mood tags, and at least one has to name the calm.
CALM_MOODS = [
    "calm", "peaceful", "serene", "still", "gentle", "soothing", "tender",
    "reflective", "spacious", "atmospheric", "warm", "mellow", "hazy", "hushed",
    "unhurried", "weightless", "meditative", "dreamlike", "laid-back", "hypnotic",
    "contemplative", "intimate", "delicate", "wistful", "melancholic", "distant",
    "nostalgic", "ethereal", "submerged", "enveloping", "vast", "deep", "dusty",
    "soulful", "cavernous",
]


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.name)
def test_every_mood_tag_is_a_calm_one(family):
    for value in family.slots["moods"]:
        for tag in (t.strip().lower() for t in value.split(",")):
            assert tag in CALM_MOODS, f"{family.name}: mood {tag!r} is not calm"


def test_negative_prompt_covers_vocals():
    assert "vocals" in NEGATIVE_PROMPT
    assert "choir" in NEGATIVE_PROMPT


def test_negative_prompt_rejects_loud_and_eventful():
    assert "aggressive" in NEGATIVE_PROMPT
    assert "heavy drums" in NEGATIVE_PROMPT
    assert "big drops" in NEGATIVE_PROMPT


def test_seed_is_reproducible():
    a, _ = build_prompts(LONG_SESSION, random.Random(42))
    b, _ = build_prompts(LONG_SESSION, random.Random(42))
    assert a == b


def test_different_seeds_give_different_playlists():
    a, _ = build_prompts(LONG_SESSION, random.Random(1))
    b, _ = build_prompts(LONG_SESSION, random.Random(2))
    assert a != b


def test_over_capacity_raises():
    with pytest.raises(ValueError, match="capacity"):
        build_prompts(capacity() + 1, random.Random(0))


def _family_of(prompt: str) -> str:
    """Recover a prompt's family from the Subgenre field, which is unique to it."""
    hits = [
        f.name for f in FAMILIES
        for subgenre in f.slots["subgenre"] if f"Subgenre: {subgenre} |" in prompt
    ]
    assert len(hits) == 1, f"{prompt!r} matched {hits}"
    return hits[0]


@pytest.mark.parametrize("seed", range(5))
def test_families_are_interleaved_not_clumped(seed):
    """No family should run many tracks deep before another cuts in."""
    prompts, _ = build_prompts(LONG_SESSION, random.Random(seed))
    labels = [_family_of(p) for p in prompts]

    longest, run = 1, 1
    for prev, cur in zip(labels, labels[1:]):
        run = run + 1 if cur == prev else 1
        longest = max(longest, run)
    assert longest <= 4, f"a family ran {longest} tracks deep"
