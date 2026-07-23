"""Combinatorial prompt-bank machinery shared by the playlist generators.

A bank is a list of `Family` objects. Each family is a sub-style with its own
slot pools, and a prompt is one value drawn from each pool, so a short
hand-written family expands into thousands of distinct prompts. Pools are sized
pairwise-coprime-ish, so cycling them odometer-style walks a long path before any
combination comes back around.

The two banks that use this -- focus_prompts.py and lifting_prompts.py -- supply
only their FAMILIES and NEGATIVE_PROMPT; all the allocation, drawing and
interleaving lives here.
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


def capacity(families: list) -> int:
    """Longest playlist the bank can fill without repeating a prompt.

    Not the sum of the families' combination counts: tracks are handed out by
    weight, so the playlist is capped by whichever family exhausts its
    combinations first -- normally the lightest-weighted one.
    """
    total = sum(f.weight for f in families)
    return min(int(f.capacity() * total / f.weight) for f in families)


def _allocate(families: list, n: int) -> dict:
    """Split n tracks across families by weight, largest remainder first."""
    total = sum(f.weight for f in families)
    exact = {f.name: n * f.weight / total for f in families}
    counts = {name: int(x) for name, x in exact.items()}

    order = sorted(families, key=lambda f: exact[f.name] - counts[f.name], reverse=True)
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


def build_prompts(n: int, families: list, rng) -> tuple:
    """Return (prompts, per-family counts): n distinct prompts, families interleaved.

    Each family's tracks are spread evenly across the playlist by sorting on a
    stratified key, so you never get four of one texture back to back.
    """
    if n > capacity(families):
        raise ValueError(f"asked for {n} prompts, bank capacity is "
                         f"{capacity(families)}")

    counts = _allocate(families, n)
    keyed = []
    for family in families:
        drawn = _draw(family, counts[family.name], rng)
        for i, prompt in enumerate(drawn):
            keyed.append(((i + rng.random()) / len(drawn), prompt))

    keyed.sort(key=lambda pair: pair[0])
    return [prompt for _, prompt in keyed], counts
