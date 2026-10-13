"""Sequence/profile/structure query searches with explicit permitted sources."""
import itertools
from bisect import bisect_right
from contextlib import ExitStack
from pathlib import Path
import shutil
import subprocess

import numpy as np

from phanet.common.io import write_json
from phanet.network.graph import parse_hmmer_domtbl


def command(args, directory, name):
    args = list(map(str, args))
    write_json(directory / (name + ".command.json"), args)
    with (directory / (name + ".log")).open("x") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)


def write_fasta(path, sequences):
    with Path(path).open("x") as stream:
        for h, seq in sequences.items():
            stream.write(f">{h}\n{seq}\n")


def database_keys(path):
    result = {}
    with Path(str(path)+".lookup").open() as stream:
        for line in stream:
            key, name, *_ = line.split()
            if name in result:
                raise ValueError("Duplicate MMseqs database identity")
            result[name] = int(key)
    return result


def audit_prefilter(path, query_keys, permitted):
    """Read MMseqs logical database offsets across physical data parts."""
    parts = sorted((p for p in path.parent.glob(path.name + ".*") if p.suffix[1:].isdigit()),
                   key=lambda p: int(p.suffix[1:]))
    if path.exists():
        if parts:
            raise ValueError("Ambiguous MMseqs database parts")
        parts = [path]
    starts = [0]
    for p in parts:
        starts.append(starts[-1]+p.stat().st_size)
    seen, counts = set(), {}
    with ExitStack() as stack:
        files = [stack.enter_context(p.open("rb")) for p in parts]
        with Path(str(path)+".index").open() as index:
            for line in index:
                key, offset, size = map(int, line.split())
                if key not in query_keys or key in seen or offset < 0 or size < 1 or offset+size > starts[-1]:
                    raise ValueError("Invalid MMseqs prefilter index")
                seen.add(key); pieces = []
                while size:
                    j = bisect_right(starts, offset)-1
                    length = min(size, starts[j+1]-offset)
                    files[j].seek(offset-starts[j]); piece = files[j].read(length)
                    if len(piece) != length or length <= 0:
                        raise ValueError("Truncated MMseqs prefilter database")
                    pieces.append(piece); offset += length; size -= length
                raw = b"".join(pieces)
                if not raw.endswith(b"\0"):
                    raise ValueError("Unterminated MMseqs prefilter record")
                targets = [int(row.split(b"\t", 1)[0]) for row in raw[:-1].splitlines()]
                if len(set(targets)) != len(targets) or not set(targets) <= permitted:
                    raise ValueError("Duplicate or forbidden prefilter source")
                counts[key] = len(targets)
    if seen != query_keys:
        raise ValueError("Incomplete MMseqs query coverage")
    return counts


def sequence_search(bundle, targets_by_contig, sequences, local_by_contig, output, *, mmseqs, threads, hidden):
    """Filter sources before prefilter top-k; use one unchanged alignment background.

    Local sources are terminal, contig-specific references. Complete target aliases
    cannot enter either the global or local source pool.
    """
    out = Path(output); out.mkdir()
    cmd = lambda name, *args: command([mmseqs, *args], out, name)
    globals_ = {bytes(h).decode(): i for i, h in enumerate(bundle.ids)}
    local = {h: sequences[h] for hs in local_by_contig.values() for h in hs if h not in globals_}
    # This FASTA contains only new visible sources; queries are never source records.
    write_fasta(out / "local.faa", dict(sorted(local.items())))
    inputs = [bundle.root / "references.faa"] + ([out / "local.faa"] if local else [])
    cmd("source-createdb", "createdb", *inputs, out / "sources", "--dbtype", 1, "--shuffle", 0, "--threads", threads)
    keys = database_keys(out / "sources")
    if set(keys) != set(globals_) | set(local):
        raise ValueError("Reference FASTA/database identities differ from resource arrays")
    all_query = {h: sequences[h] for hs in targets_by_contig.values() for h in hs}
    write_fasta(out / "queries.faa", dict(sorted(all_query.items())))
    cmd("query-createdb", "createdb", out / "queries.faa", out / "queries", "--dbtype", 1, "--shuffle", 0, "--threads", threads)
    qkeys = database_keys(out / "queries")
    # Compute actual diagonal alignment scores for both queries and new local sources.
    self_seqs = dict(sorted({**all_query, **local}.items()))
    write_fasta(out / "self.faa", self_seqs)
    cmd("self-createdb", "createdb", out / "self.faa", out / "self", "--dbtype", 1, "--shuffle", 0, "--threads", threads)
    skeys = database_keys(out / "self")
    (out / "diagonal.tsv").write_text("".join(f"{k}\t{k}\t0\t0\n" for k in sorted(skeys.values())))
    cmd("self-pref", "tsv2db", out / "diagonal.tsv", out / "diagonal", "--output-dbtype", 7)
    cmd("self-align", "align", out / "self", out / "self", out / "diagonal", out / "self-aln",
        "-e", "1e30", "--alignment-mode", 3, "-a", 1, "--threads", threads)
    cmd("self-convert", "convertalis", out / "self", out / "self", out / "self-aln", out / "self.tsv",
        "--format-output", "query,target,bits", "--threads", threads)
    selfbits = {}
    for line in (out / "self.tsv").read_text().splitlines():
        q, r, bits = line.split(); value = float(bits)
        if q != r or q in selfbits or not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid query/source self-alignment")
        selfbits[q] = value
    if set(selfbits) != set(self_seqs):
        raise ValueError("Incomplete self-alignment scores")
    result = {}
    for group, (contig, targets) in enumerate(targets_by_contig.items()):
        permitted = (set(globals_) | set(local_by_contig[contig])) - hidden
        (out / f"sources-{group}.keys").write_text("".join(f"{keys[h]}\n" for h in sorted(permitted, key=keys.get)))
        (out / f"queries-{group}.keys").write_text("".join(f"{qkeys[h]}\n" for h in sorted(set(targets), key=qkeys.get)))
        result[contig] = {h: {"hits": [], "edges": {}} for h in targets}
        if not permitted:
            continue
        source_db, query_db = out / f"s-{group}", out / f"q-{group}"
        cmd(f"source-sub-{group}", "createsubdb", out / f"sources-{group}.keys", out / "sources", source_db, "--subdb-mode", 1)
        cmd(f"query-sub-{group}", "createsubdb", out / f"queries-{group}.keys", out / "queries", query_db, "--subdb-mode", 1)
        cmd(f"prefilter-{group}", "prefilter", query_db, source_db, out / f"pref-{group}", "-s", "7.5", "-k", 6,
            "--max-seqs", len(permitted), "--threads", threads, "--split-memory-limit", "8G")
        # The budget covers all permitted targets, so no prefilter top-k truncation
        # can change the best bit-score votes or normalized sequence neighbors.
        counts = audit_prefilter(out / f"pref-{group}", {qkeys[h] for h in targets}, {keys[h] for h in permitted})
        write_json(out / f"retrieval-{group}.json", {"counts": counts, "permitted_sources": len(permitted)})
        cmd(f"align-{group}", "align", query_db, out / "sources", out / f"pref-{group}", out / f"aln-{group}",
            "-e", "1e-5", "--alignment-mode", 3, "-a", 1, "--threads", threads)
        table = out / f"hits-{group}.tsv"
        cmd(f"convert-{group}", "convertalis", query_db, out / "sources", out / f"aln-{group}", table,
            "--format-output", "query,target,fident,evalue,bits,qcov,tcov", "--threads", threads)
        seen = set()
        with table.open() as stream:
            for line in stream:
                q, r, ident, ev, bits, qc, rc = line.split()
                if q not in result[contig] or r not in permitted or (q, r) in seen:
                    raise ValueError("Duplicate or non-permitted sequence hit")
                seen.add((q, r))
                ident, ev, bits, qc, rc = map(float, (ident, ev, bits, qc, rc))
                if (not np.isfinite([ident, ev, bits, qc, rc]).all() or ev < 0 or bits < 0
                        or not 0 <= min(ident, qc, rc) <= max(ident, qc, rc) <= 1):
                    raise ValueError("Invalid sequence hit quality")
                if ev > 1e-5 or bits < 50:
                    continue
                result[contig][q]["hits"].append(dict(reference_id=r, bits=bits, evalue=ev, identity=ident,
                                                     query_coverage=qc, reference_coverage=rc))
                rb = bundle.self_bits[globals_[r]] if r in globals_ else selfbits[r]
                score = bits / max(selfbits[q], rb)
                if score > 1+1e-6:
                    raise ValueError("Alignment score exceeds its self-score normalization")
                if score >= .3:
                    result[contig][q]["edges"][r] = min(1., score)
        for entry in result[contig].values():
            entry["edges"] = dict(sorted(entry["edges"].items(), key=lambda x: (-x[1], x[0]))[:200])
    return result


def profile_search(sequences, profiles, output, *, threads, progress=print,
                   profile_count=38880, profile_Z=532910):
    import pyhmmer
    out = Path(output); out.mkdir()
    write_fasta(out / "proteins.faa", dict(sorted(sequences.items())))
    with pyhmmer.easel.SequenceFile(str(out / "proteins.faa"), digital=True, alphabet=pyhmmer.easel.Alphabet.amino()) as handle:
        proteins = handle.read_block()
    names = set()
    # Large buffered reads avoid a filesystem round trip for every small HMM
    # parser read, especially on mounted drives. The model bytes are unchanged.
    with Path(profiles).open("rb", buffering=8*1024**2) as source, \
            pyhmmer.plan7.HMMFile(source) as models, (out / "domains.tsv").open("xb") as raw:
        while batch := list(itertools.islice(models, 64)):
            for model in batch:
                if model.name in names:
                    raise ValueError("Duplicate profile family")
                names.add(model.name)
            count = 0
            for hits in pyhmmer.hmmer.hmmsearch(batch, proteins, cpus=threads, backend="threading",
                                               E=1e-5, domE=1e-5, seed=42, Z=profile_Z):
                hits.write(raw, format="domains", header=False); count += 1
            if count != len(batch):
                raise ValueError("Incomplete profile search batch")
            if len(names) % 4096 == 0:
                progress(f"Protein profiles: {len(names)}/{profile_count}")
    if len(names) != profile_count:
        raise ValueError(f"Expected {profile_count} protein profiles")
    hits = parse_hmmer_domtbl(out / "domains.tsv", mode="hmmsearch", evalue=1e-5, minimum_coverage=.5)
    write_json(out / "hits.json", hits)
    return hits


def profile_edges(bundle, query_hits, local_hits, allowed, local_ids):
    """Per-family top 50 by source coverage, then complete sequence hash."""
    rows = {}
    for hit in query_hits:
        family, cov = hit["family_id"], hit["coverage"]
        candidates = []
        if family in bundle.family_index:
            i = bundle.family_index[family]
            a, b = bundle.profile_indptr[i:i+2]
            ix = bundle.profile_indices[a:b]; cc = bundle.profile_coverage[a:b]
            keep = allowed[ix]
            candidates.extend((bytes(bundle.ids[j]).decode(), float(c)) for j, c in zip(ix[keep], cc[keep]))
        candidates.extend((h["protein_id"], h["coverage"]) for h in local_hits
                          if h["family_id"] == family and h["protein_id"] in local_ids)
        candidates.sort(key=lambda x: (-x[1], x[0]))
        row = rows.setdefault(hit["protein_id"], {})
        for h, coverage in candidates[:50]:
            row[h] = max(row.get(h, 0), min(cov, coverage))
    return rows


def structure_edges(bundle, query_vectors, local_vectors, allowed, *, block=32768):
    """Exact blockwise cosine search; deterministic hash ordering for ties."""
    result = {}
    for q, vector in query_vectors.items():
        best = []
        for lo in range(0, len(bundle.ids), block):
            hi = min(lo+block, len(bundle.ids))
            scores = bundle.vectors[lo:hi] @ vector
            ix = np.flatnonzero(allowed[lo:hi] & (scores >= .5))
            # Lexsort within each block also handles ties at the 200th position.
            ix = ix[np.lexsort((bundle.ids[lo:hi][ix], -scores[ix]))[:200]]
            best.extend((bytes(bundle.ids[lo+j]).decode(), min(1., float(scores[j]))) for j in ix)
            best = sorted(best, key=lambda x: (-x[1], x[0]))[:200]
        best.extend((h, min(1., float(v @ vector))) for h, v in local_vectors.items() if v @ vector >= .5)
        result[q] = dict(sorted(best, key=lambda x: (-x[1], x[0]))[:200])
    return result
