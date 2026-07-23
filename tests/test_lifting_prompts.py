"""Tests for the lifting playlist prompt bank (lifting_prompts.py).

The bank's promise is "thousands of distinct prompts, none of which sing, all of
which are four-on-the-floor". Each half is easy to break by hand-editing a pool,
so these are pure data checks -- no model, no torch, no weights.
"""

import random
import re
import string
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lifting_prompts import (  # noqa: E402
    FAMILIES,
    NEGATIVE_PROMPT,
    build_prompts,
    capacity,
)

# A long lifting session at the 120s per-track ceiling of the small-music model:
# four hours. The bank's real capacity is many times this; the point of the test
# is that no realistic session ever repeats a prompt.
LONG_SESSION = 120

# Anything that might invite a vocal take. Matched on word boundaries so
# "lyrical" and "rising" don't trip "lyrics" and "sing".
VOCAL_WORDS = [
    "vocal", "vocals", "voice", "voices", "choir", "choral", "sing", "singing",
    "sung", "lyrics", "lyric", "chant", "chanting", "spoken", "speech",
    "acapella", "vox", "soprano", "tenor", "baritone", "humming",
]
VOCAL_RE = re.compile(rf"\b({'|'.join(VOCAL_WORDS)})\b", re.IGNORECASE)

# Grooves that pull against strict four-on-the-floor. If one of these leaks into
# a slot, the steady pulse you rep to is gone.
OFFBEAT_WORDS = [
    "breakbeat", "amen", "drum and bass", "half-time", "halftime", "breakcore",
    "dubstep", "two-step",
]
OFFBEAT_RE = re.compile(rf"\b({'|'.join(w for w in OFFBEAT_WORDS if ' ' not in w)})\b",
                        re.IGNORECASE)


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
# Generated prompts
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


def test_every_prompt_forbids_vocals():
    prompts, _ = build_prompts(LONG_SESSION, random.Random(0))
    assert all(p.endswith("no vocals") or ", no vocals, " in p for p in prompts)


def test_no_slot_value_invites_a_vocal():
    for name, slot, value in _slot_values():
        match = VOCAL_RE.search(value)
        assert not match, f"{name}.{slot}: {value!r} contains {match.group(0)!r}"


# A motion slot has to read as a steady four-on-the-floor, but it can say so in
# several ways -- naming the kick, "4/4", "four-on-the-floor", or the beat.
FOUR_ON_FLOOR_MARKERS = ["kick", "four-on-the-floor", "4/4", "every beat"]


def test_every_motion_is_four_on_the_floor():
    """The whole bank exists to keep a steady pulse under every rep."""
    for name, slot, value in _slot_values():
        if slot == "motion":
            low = value.lower()
            assert any(m in low for m in FOUR_ON_FLOOR_MARKERS), (
                f"{name}.motion: {value!r} names no four-on-the-floor marker"
            )


def test_no_slot_value_breaks_the_pulse():
    """No breakbeat/half-time groove should sneak in to fight the 4/4."""
    for name, slot, value in _slot_values():
        assert "drum and bass" not in value.lower(), f"{name}.{slot}: {value!r}"
        match = OFFBEAT_RE.search(value)
        assert not match, f"{name}.{slot}: {value!r} contains {match.group(0)!r}"


def test_negative_prompt_covers_vocals():
    assert "vocals" in NEGATIVE_PROMPT
    assert "choir" in NEGATIVE_PROMPT


def test_negative_prompt_rejects_offbeat_grooves():
    assert "breakbeat" in NEGATIVE_PROMPT
    assert "drum and bass" in NEGATIVE_PROMPT


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
    """Recover a prompt's family from its opening style clause."""
    hits = [
        f.name for f in FAMILIES
        for style in f.slots["style"] if prompt.startswith(style + ",")
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
