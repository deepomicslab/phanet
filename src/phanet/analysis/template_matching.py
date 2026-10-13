"""Qualified profile and directed sequence evidence for module matching."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ProfileHit:
    profile: str
    evalue: float
    coverage: float

    def __post_init__(self):
        if (not self.profile or not math.isfinite(self.evalue) or self.evalue < 0
                or not math.isfinite(self.coverage) or not 0 <= self.coverage <= 1):
            raise ValueError("Invalid profile evidence")

    @property
    def eligible(self):
        return self.evalue <= 1e-5 and self.coverage >= .5


@dataclass(frozen=True)
class SequenceHit:
    evalue: float
    bits: float
    query_self_bits: float
    target_self_bits: float
    query_coverage: float
    target_coverage: float

    def __post_init__(self):
        values = (self.evalue, self.bits, self.query_self_bits, self.target_self_bits,
                  self.query_coverage, self.target_coverage)
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError("Invalid sequence evidence")
        if min(self.query_self_bits, self.target_self_bits) <= 0 or max(self.query_coverage, self.target_coverage) > 1:
            raise ValueError("Invalid sequence coverage/self scores")

    @property
    def normalized_bits(self):
        return self.bits / max(self.query_self_bits, self.target_self_bits)

    @property
    def eligible(self):
        return (self.evalue <= 1e-5 and self.bits >= 50 and self.normalized_bits >= .3
                and min(self.query_coverage, self.target_coverage) >= .5)


class Relationships:
    """Index protein relationships by sequence hash, profile and directed alignment."""

    def __init__(self, profiles=None, sequences=None, auxiliary=None):
        self.profiles = {p: frozenset(h.profile for h in hits if h.eligible)
                         for p, hits in (profiles or {}).items()}
        self.sequences = dict(sequences or {})
        self.auxiliary = dict(auxiliary or {})
        if any(not math.isfinite(x) for x in self.auxiliary.values()):
            raise ValueError("Nonfinite auxiliary score")

    def score(self, left, right):
        profile = bool(self.profiles.get(left, frozenset()) & self.profiles.get(right, frozenset()))
        # The lookup is directed so E-values retain their actual search universe.
        seq = self.sequences.get((left, right))
        sequence_ok = seq is not None and seq.eligible
        if not profile and not sequence_ok:
            return None
        auxiliary = self.auxiliary.get((left, right), 0.)
        return (int(profile), seq.normalized_bits if sequence_ok else 0., auxiliary)
