"""Directed sequence-alignment validation and score normalization."""
from __future__ import annotations

import csv
from pathlib import Path
import sqlite3

from phanet.analysis.template_matching import SequenceHit


def read_pilot_alignments(path: str | Path, proteins):
    """Read query, target, E-value, bits and coverage columns with measured self scores."""
    selected = set(proteins)
    if not selected:
        raise ValueError("Empty sequence search universe")
    rows = {}
    self_bits = {}
    with Path(path).open(newline="") as handle:
        for line, fields in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if len(fields) != 6:
                raise ValueError(f"Expected six alignment fields at row {line}")
            query, target = fields[:2]
            if query not in selected or target not in selected:
                raise ValueError(f"Alignment outside search universe at row {line}")
            if (query, target) in rows:
                raise ValueError(f"Duplicate directed alignment at row {line}")
            evalue, bits, qcov, tcov = map(float, fields[2:])
            SequenceHit(evalue, bits, 1., 1., qcov, tcov)
            rows[query, target] = (evalue, bits, qcov, tcov)
            if query == target:
                if bits <= 0 or qcov != 1. or tcov != 1.:
                    raise ValueError(f"Invalid full self alignment at row {line}")
                self_bits[query] = bits
    if set(self_bits) != selected:
        raise ValueError(f"Missing measured self scores: {len(selected - self_bits.keys())}")
    relations = {
        (query, target): SequenceHit(evalue, bits, self_bits[query], self_bits[target], qcov, tcov)
        for (query, target), (evalue, bits, qcov, tcov) in rows.items()
    }
    return relations, self_bits


class DiskAlignments:
    """Validate directed alignment rows in SQLite and stream normalized scores."""

    def __init__(self, path, proteins, database):
        self.proteins = list(proteins)
        ids = {p: i for i, p in enumerate(self.proteins)}
        if not ids or len(ids) != len(self.proteins):
            raise ValueError("Empty or duplicate sequence search universe")
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.database.open("xb"):
            pass
        n = len(ids)
        if n * n > 2**63 - 1:
            raise ValueError("Sequence universe exceeds integer key capacity")
        self.self_bits = {}
        self.count = 0
        self.db = sqlite3.connect(self.database)
        try:
            self.db.execute("PRAGMA cache_size=-32768")
            self.db.execute("PRAGMA temp_store=FILE")
            self.db.execute("CREATE TABLE hits (pair INTEGER PRIMARY KEY, evalue REAL, bits REAL, qcov REAL, tcov REAL)")
            batch = []
            with Path(path).open(newline="") as stream:
                for line, fields in enumerate(csv.reader(stream, delimiter="\t"), 1):
                    if len(fields) != 6:
                        raise ValueError(f"Expected six alignment fields at row {line}")
                    q, t = fields[:2]
                    if q not in ids or t not in ids:
                        raise ValueError(f"Alignment outside search universe at row {line}")
                    e, bits, qc, tc = map(float, fields[2:])
                    SequenceHit(e, bits, 1., 1., qc, tc)
                    if q == t:
                        if bits <= 0 or qc != 1. or tc != 1.:
                            raise ValueError(f"Invalid full self alignment at row {line}")
                        self.self_bits[q] = bits
                    batch.append((ids[q] * n + ids[t], e, bits, qc, tc))
                    self.count += 1
                    if len(batch) == 10000:
                        self.db.executemany("INSERT INTO hits VALUES (?,?,?,?,?)", batch)
                        self.db.commit()
                        batch.clear()
            self.db.executemany("INSERT INTO hits VALUES (?,?,?,?,?)", batch)
            self.db.commit()
            if self.self_bits.keys() != ids.keys():
                raise ValueError(f"Missing measured self scores: {len(ids.keys() - self.self_bits.keys())}")
            self.db.execute("PRAGMA query_only=ON")
        except BaseException as exc:
            self.close()
            if isinstance(exc, sqlite3.IntegrityError):
                raise ValueError("Duplicate directed alignment") from exc
            raise

    def items(self):
        n = len(self.proteins)
        for pair, e, bits, qc, tc in self.db.execute("SELECT pair,evalue,bits,qcov,tcov FROM hits ORDER BY pair"):
            qi, ti = divmod(pair, n)
            q, t = self.proteins[qi], self.proteins[ti]
            yield (q, t), SequenceHit(e, bits, self.self_bits[q], self.self_bits[t], qc, tc)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
