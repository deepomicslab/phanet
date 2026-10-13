"""Portable, checksummed resources for three-channel network inference."""
from pathlib import Path

import numpy as np
from scipy import sparse

from phanet.common.constants import CATEGORIES
from phanet.common.io import digest_file, read_json


ARRAYS = ("ids", "labels", "self_bits", "vectors", "support", "graph_data",
          "graph_indices", "graph_indptr", "profile_indices", "profile_indptr", "profile_coverage")


class NetworkBundle:
    """Load checksummed reference arrays with memory mapping."""

    def __init__(self, directory, *, progress=print):
        self.root = Path(directory).expanduser().resolve()
        self.meta = read_json(self.root / "network.json")
        m = self.meta
        if (m.get("schema_version") not in (1, 2) or m.get("categories") != list(CATEGORIES)
                or m.get("method") != "three-channel-fixed-label-ppr-v1"
                or m.get("embedding") != "prott5-tmvec:full_length:float16:l2"
                or not isinstance(m.get("profile_Z"), int) or m["profile_Z"] < 1
                or not isinstance(m.get("profile_count"), int) or m["profile_count"] < 1
                or (m["schema_version"] == 1 and (m["profile_Z"] != 532910 or m["profile_count"] != 38880))):
            raise ValueError("Unsupported network resource contract")
        required = {n + ".npy" for n in ARRAYS} | {"families.json", "references.faa"}
        if set(m.get("files", {})) != required:
            raise ValueError("Incomplete or unexpected network bundle files")
        for name, record in m["files"].items():
            path = self.root / name
            progress(f"Verifying network resource: {name} ({record['bytes'] / 1e9:.2f} GB)")
            if path.is_symlink() or path.stat().st_size != record["bytes"] or digest_file(path) != record["sha256"]:
                raise ValueError(f"Network resource integrity failure: {name}")
        progress("Network resource checksums verified")
        for name in ARRAYS:
            setattr(self, name, np.load(self.root / (name + ".npy"), mmap_mode="r", allow_pickle=False))
        n = len(self.ids)
        if (not n or self.ids.dtype != np.dtype("S64") or self.ids.shape != (n,)
                or not np.all(self.ids[:-1] < self.ids[1:])
                or self.labels.shape != (n,) or self.labels.dtype.kind not in "ui"
                or np.any(self.labels > 511) or self.self_bits.shape != (n,)
                or not np.isfinite(self.self_bits).all() or np.any(self.self_bits <= 0)
                or self.vectors.shape != (n, 512) or self.support.shape != (n, 10)):
            raise ValueError("Invalid reference identities, labels or feature dimensions")
        for h in self.ids:
            if len(h) != 64 or any(c not in b"0123456789abcdef" for c in h):
                raise ValueError("References must use complete sequence SHA256 identities")
        for lo in range(0, n, 16384):
            vec = self.vectors[lo:lo+16384]
            sup = self.support[lo:lo+16384]
            if (not np.isfinite(vec).all() or not np.allclose(np.linalg.norm(vec, axis=1), 1, atol=2e-6)
                    or not np.isfinite(sup).all() or np.any((sup < 0) | (sup > 1+1e-12))
                    or np.any(sup[:, :9] > sup[:, 9, None]+1e-12)):
                raise ValueError("Invalid reference vectors or PPR masses")
        self.families = read_json(self.root / "families.json")
        if not isinstance(self.families, list) or len(set(self.families)) != len(self.families):
            raise ValueError("Invalid profile family identities")
        for ptr, ix, values, rows in ((self.graph_indptr, self.graph_indices, self.graph_data, n),
                                     (self.profile_indptr, self.profile_indices, self.profile_coverage, len(self.families))):
            if (ptr.shape != (rows+1,) or ptr.dtype.kind not in "ui" or ix.dtype.kind not in "ui"
                    or ix.ndim != 1 or values.shape != ix.shape or ptr[0] != 0 or ptr[-1] != len(ix)
                    or np.any(ptr[1:] < ptr[:-1])):
                raise ValueError("Invalid sparse resource offsets")
            for lo in range(0, len(ix), 2_000_000):
                j, v = ix[lo:lo+2_000_000], values[lo:lo+2_000_000]
                if np.any(j < 0) or np.any(j >= n) or not np.isfinite(v).all() or np.any((v <= 0) | (v > 1)):
                    raise ValueError("Invalid sparse resource entries")
        self.family_index = {f: i for i, f in enumerate(self.families)}

    def positions(self, hashes):
        values = np.asarray(hashes, dtype="S64")
        pos = np.searchsorted(self.ids, values)
        found = pos < len(self.ids)
        found[found] &= self.ids[pos[found]] == values[found]
        return np.where(found, pos, -1)

    def label_matrix(self):
        return ((self.labels[:, None] >> np.arange(9)) & 1).astype(np.float64)

    def masked_support(self, hidden, *, progress=print, chunk_rows=4096):
        """Recompute diffusion on the induced graph if a target aliases a reference.

        Streaming sparse blocks bound memory without changing fixed-label PPR.
        The released precomputed support is used only when no reference is hidden.
        """
        allowed = np.ones(len(self.ids), bool)
        pos = self.positions(sorted(hidden)); allowed[pos[pos >= 0]] = False
        if allowed.all():
            return self.support, allowed, {"cached": True, "hidden_references": 0}
        seed = np.column_stack([self.label_matrix(), self.labels != 0])
        seed[~allowed] = 0
        old = seed.copy()
        degree = np.zeros(len(allowed), np.float64)
        for lo in range(0, len(allowed), chunk_rows):
            hi = min(lo+chunk_rows, len(allowed))
            mat = self.graph_block(lo, hi)
            degree[lo:hi] = mat @ allowed.astype(np.float64)
        for iteration in range(1, 201):
            new = np.empty_like(old)
            for lo in range(0, len(allowed), chunk_rows):
                hi = min(lo+chunk_rows, len(allowed))
                walk = self.graph_block(lo, hi) @ old
                d = degree[lo:hi, None]
                new[lo:hi] = .15*seed[lo:hi] + .85*np.divide(walk, d, out=np.zeros_like(walk), where=d > 0)
            new[degree == 0] = seed[degree == 0]
            new[~allowed] = 0
            residual = float(np.max(np.abs(new-old)))
            old = new
            progress(f"Reference PPR: iteration {iteration}, residual {residual:.3g}")
            if residual < 1e-7:
                return old, allowed, {"cached": False, "hidden_references": int((~allowed).sum()),
                                      "iterations": iteration, "residual": residual}
        raise RuntimeError("Masked reference PPR failed to converge")

    def graph_block(self, lo, hi):
        a, b = int(self.graph_indptr[lo]), int(self.graph_indptr[hi])
        return sparse.csr_matrix((self.graph_data[a:b], self.graph_indices[a:b],
                                  self.graph_indptr[lo:hi+1]-a), shape=(hi-lo, len(self.ids)))
